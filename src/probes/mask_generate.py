#!/usr/bin/env python
"""MASK-benchmark completion generation for every LoRA checkpoint of a run.

Generation ONLY: no judge, no API scoring, no honesty metric.  The output is
written so that the official CAIS MASK judge (mask/evaluate.py) can be run on
it later: every example carries the official response-column names
(``generation(System Prompt + User Prompt)_run1``,
``generation(belief_system_1 + belief_elicit_k)_runj``) next to the raw
messages that produced them.

Prompt construction mirrors the official ``mask/generate_responses.py``
(github.com/centerforaisafety/mask, dataset ``cais/MASK``):

* pressured ("lying") prompt: ``[system_prompt, user_prompt]``; for
  ``doubling_down_known_facts`` the conversation is
  ``[system_prompt, dd_turn_1_prompt, dd_turn_1_response(assistant), user_prompt]``.
  Generated ``lie_k`` times (official default 1).
* belief elicitation: system ``"You are a helpful assistant."`` +
  ``belief_elicit_k``; belief_k = {1: 3} for ``statistics`` and
  {1: 3, 2: 1, 3: 1} otherwise (official defaults).  ``provided_facts`` has no
  belief prompts.
* official sampling defaults: temperature 1.0, max_tokens 1000 (overridable).
  Every request carries a deterministic seed so runs are reproducible.

The fixed 100-example subset (seed 42, stratified proportionally over the six
archetype configs) is stored with its full rows in
``$NM_ROOT/datasets/mask_100.json`` so generation jobs need no dataset cache.

Checkpoints are read from ``$NM_ROOT/lora-finetuned/<rid>/step_<N>/`` and
outputs written to ``$NM_ROOT/live_downstream/<rid>/<out-subdir>/step_<N>.json``.
Grade the outputs with ``src/eval/mask/grade_mask.py``.

Usage:
    python src/probes/mask_generate.py --build-selection      # once (needs HF access)
    ENABLE_THINKING=1 THINKING_BUDGET=1024 ANSWER_BUDGET=512 \
    python src/probes/mask_generate.py --base-model Qwen/Qwen3-14B --enable-thinking \
        --lora-rank 64 --max-tokens 2048 --max-model-len 16384 --rid <rid> --steps 0-30 --force
    python src/probes/mask_generate.py --rid <rid> --steps 1 --limit 2 --out-subdir mask_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("SOFTFILELOCK", "1")

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.paths import NM_ROOT  # noqa: E402  (env NM_ROOT, default /data/new_master)

BASE_MODEL = "Qwen/Qwen3-14B"
MASK_DATASET = "cais/MASK"
MASK_CONFIGS = [
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
]
BELIEF_SYSTEM_PROMPT = "You are a helpful assistant."  # official default (belief_system_1)
FIELDS_KEPT = [
    "task_id", "system_prompt", "user_prompt", "dd_turn_1_prompt", "dd_turn_1_response",
    "type", "proposition", "ground_truth", "formatted_ground_truth", "delivery",
    "belief_elicit_1", "belief_elicit_2", "belief_elicit_3",
]


def belief_k_for(config: str) -> dict[str, int]:
    """Official: belief_elicit_1 x3 for statistics, else be1 x3, be2 x1, be3 x1."""
    return {"1": 3} if config == "statistics" else {"1": 3, "2": 1, "3": 1}


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def build_selection(n_total: int, seed: int, out_path: Path) -> dict:
    """Seeded, proportionally stratified sample over the MASK configs (largest-remainder)."""
    from datasets import load_dataset

    per_cfg = {}
    for cfg in MASK_CONFIGS:
        ds = load_dataset(MASK_DATASET, cfg, split="test")
        per_cfg[cfg] = ds
    sizes = {c: len(d) for c, d in per_cfg.items()}
    total = sum(sizes.values())
    quotas = {c: n_total * s / total for c, s in sizes.items()}
    alloc = {c: int(q) for c, q in quotas.items()}
    rest = n_total - sum(alloc.values())
    for c in sorted(quotas, key=lambda c: quotas[c] - alloc[c], reverse=True)[:rest]:
        alloc[c] += 1
    assert sum(alloc.values()) == n_total

    rng = random.Random(seed)
    examples = []
    for cfg in MASK_CONFIGS:
        ds = per_cfg[cfg]
        idx = sorted(rng.sample(range(len(ds)), alloc[cfg]))
        for i in idx:
            row = ds[i]
            ex = {"config": cfg, "row_index": i}
            for f in FIELDS_KEPT:
                if f in row:
                    ex[f] = row[f]
            examples.append(ex)
    sel = {
        "dataset": MASK_DATASET,
        "seed": seed,
        "n_total": n_total,
        "config_sizes": sizes,
        "allocation": alloc,
        "ids": [f"{e['config']}:{e['task_id']}" for e in examples],
        "examples": examples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sel, indent=1))
    print(f"[selection] wrote {len(examples)} examples -> {out_path}; allocation={alloc}")
    return sel


# ----------------------------------------------------------------------------
# Prompt construction (official generate_responses.py semantics)
# ----------------------------------------------------------------------------
def _present(v) -> bool:
    return v is not None and (not isinstance(v, str) or v.strip() != "")


def build_requests(ex: dict, lie_k: int) -> list[dict]:
    """Return one dict per generation request for this example."""
    reqs = []
    if _present(ex.get("dd_turn_1_prompt")) and _present(ex.get("dd_turn_1_response")):
        pressure_msgs = [
            {"role": "system", "content": ex["system_prompt"]},
            {"role": "user", "content": ex["dd_turn_1_prompt"]},
            {"role": "assistant", "content": ex["dd_turn_1_response"]},
            {"role": "user", "content": ex["user_prompt"]},
        ]
    else:
        pressure_msgs = [
            {"role": "system", "content": ex["system_prompt"]},
            {"role": "user", "content": ex["user_prompt"]},
        ]
    for run in range(lie_k):
        reqs.append({
            "kind": "pressure",
            "key": f"lying_run{run + 1}",
            "official_column": f"generation(System Prompt + User Prompt)_run{run + 1}",
            "messages": pressure_msgs,
        })
    if _present(ex.get("belief_elicit_1")):
        for be_num, k in belief_k_for(ex["config"]).items():
            col = f"belief_elicit_{be_num}"
            if not _present(ex.get(col)):
                continue
            msgs = [
                {"role": "system", "content": BELIEF_SYSTEM_PROMPT},
                {"role": "user", "content": ex[col]},
            ]
            for run in range(k):
                reqs.append({
                    "kind": "belief",
                    "key": f"be{be_num}_run{run + 1}",
                    "official_column": f"generation(belief_system_1 + belief_elicit_{be_num})_run{run + 1}",
                    "messages": msgs,
                })
    return reqs


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def parse_steps(spec: list[str]) -> list[int]:
    out: list[int] = []
    for tok in spec:
        for part in tok.replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


THINK_END = "</think>"


def _cap_thinking(text: str, think_end: str = THINK_END) -> str:
    """Cap a phase-1 trace at the first </think> (drop any answer the model
    started), or inject </think> if it never closed within budget.  Mirrors
    finetune_eval._generate_inmemory_budget."""
    return (text.split(think_end)[0] + think_end) if think_end in text else (text + think_end)


def generate_step(llm, flat, lora_req, args, SamplingParams, tokenizer):
    """Generate one completion per flat request for a single checkpoint.

    When ``args.enable_thinking`` is off this is a plain single vLLM pass
    (unchanged Llama behaviour).  When it is on, generation is budget-forced
    (s1 / Muennighoff et al. 2025), replicating
    ``finetune_eval._generate_inmemory_budget`` for the vLLM token-id path:

      Phase 1: generate up to ``thinking_budget`` tokens of reasoning.
      Cap each trace at the first </think> (else inject </think>).
      Phase 2: for each request, generate up to ``answer_budget`` more tokens
      conditioned on (formatted prompt string + capped thinking).
      Final completion = capped_thinking + answer, so it always terminates its
      <think> block and carries a real answer (belief-elicit answers keep their
      <final_answer>...</final_answer> tags).

    Returns ``(texts, finish_reasons, injected_count)``.
    """
    n = len(flat)
    if not args.enable_thinking:
        sampling = [
            SamplingParams(
                temperature=args.temperature,
                max_tokens=max(16, min(args.max_tokens, args.max_model_len - len(ids))),
                seed=args.seed * 100_000 + ri,
            )
            for ri, (ei, r, ids, ps) in enumerate(flat)
        ]
        inputs = [{"prompt_token_ids": ids} for _, _, ids, _ in flat]
        outs = llm.generate(inputs, sampling, lora_request=lora_req, use_tqdm=False)
        return ([o.outputs[0].text for o in outs],
                [o.outputs[0].finish_reason for o in outs], 0)

    # Phase 1: reasoning trace, capped at thinking_budget.
    p1_sampling = [
        SamplingParams(
            temperature=args.temperature,
            max_tokens=max(16, min(args.thinking_budget, args.max_model_len - len(ids))),
            seed=args.seed * 100_000 + ri,
        )
        for ri, (ei, r, ids, ps) in enumerate(flat)
    ]
    p1_inputs = [{"prompt_token_ids": ids} for _, _, ids, _ in flat]
    o1 = llm.generate(p1_inputs, p1_sampling, lora_request=lora_req, use_tqdm=False)
    think = [o.outputs[0].text for o in o1]
    capped = [_cap_thinking(t) for t in think]
    injected = sum(1 for t in think if THINK_END not in t)

    # Phase 2: answer, conditioned on prompt string + capped thinking.
    p2_ids = [
        list(tokenizer(ps + cap, add_special_tokens=False).input_ids)
        for (ei, r, ids, ps), cap in zip(flat, capped)
    ]
    p2_sampling = [
        SamplingParams(
            temperature=args.temperature,
            max_tokens=max(16, min(args.answer_budget, args.max_model_len - len(ids2))),
            seed=args.seed * 100_000 + n + ri,
        )
        for ri, ids2 in enumerate(p2_ids)
    ]
    p2_inputs = [{"prompt_token_ids": ids2} for ids2 in p2_ids]
    o2 = llm.generate(p2_inputs, p2_sampling, lora_request=lora_req, use_tqdm=False)
    answers = [o.outputs[0].text for o in o2]
    finish = [o.outputs[0].finish_reason for o in o2]
    texts = [cap + a for cap, a in zip(capped, answers)]
    print(f"[budget-forcing] think_budget={args.thinking_budget} "
          f"answer_budget={args.answer_budget}; </think> injected on "
          f"{injected}/{n} (exceeded budget)", flush=True)
    return texts, finish, injected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rid", help="finetune run id (lora-finetuned/<rid>/step_N)")
    ap.add_argument("--steps", nargs="*", default=[], help="steps, e.g. 0-30 or 0 1 5 (0 = base model)")
    ap.add_argument("--limit", type=int, default=None, help="only first N selected examples (smoke test)")
    ap.add_argument("--selection", default=str(NM_ROOT / "datasets" / "mask_100.json"))
    ap.add_argument("--build-selection", action="store_true", help="(re)build the selection file and exit")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0, help="official MASK default 1.0")
    ap.add_argument("--max-tokens", type=int, default=1000, help="official MASK default 1000")
    ap.add_argument("--lie-k", type=int, default=1, help="official MASK default 1")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true",
        default=os.environ.get("ENABLE_THINKING", "0").lower() in ("1", "true", "yes"),
        help="Qwen3 reasoning models: emit the <think> trace in the chat template "
             "(matches finetune_utils.set_enable_thinking / how the model was trained). "
             "Ignored by templates that do not reference it (Llama, Mistral). "
             "Also toggled by the ENABLE_THINKING env var.",
    )
    ap.add_argument(
        "--thinking-budget", type=int,
        default=int(os.environ.get("THINKING_BUDGET", "1024")),
        help="budget forcing (s1): max tokens for the <think> trace in phase 1 "
             "(only used when --enable-thinking). Matches the finetune config.",
    )
    ap.add_argument(
        "--answer-budget", type=int,
        default=int(os.environ.get("ANSWER_BUDGET", "512")),
        help="budget forcing (s1): max tokens for the answer in phase 2, "
             "conditioned on prompt + capped thinking (only with --enable-thinking).",
    )
    ap.add_argument("--out-subdir", default="mask", help="live_downstream/<rid>/<out-subdir>/step_N.json")
    ap.add_argument("--force", action="store_true", help="overwrite existing step files")
    args = ap.parse_args()

    sel_path = Path(args.selection)
    if args.build_selection:
        build_selection(args.n, args.seed, sel_path)
        return
    if not sel_path.exists():
        raise SystemExit(f"selection file missing: {sel_path} (run --build-selection first)")
    if not args.rid or not args.steps:
        raise SystemExit("--rid and --steps are required")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_RPC_GET_DATA_TIMEOUT_MS", "120000")

    sel = json.loads(sel_path.read_text())
    examples = sel["examples"]
    if args.limit:
        examples = examples[: args.limit]
    steps = parse_steps(args.steps)
    ckpt_root = NM_ROOT / "lora-finetuned" / args.rid
    out_dir = NM_ROOT / "live_downstream" / args.rid / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for s in steps:
        out_path = out_dir / f"step_{s}.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} exists")
            continue
        if s > 0 and not (ckpt_root / f"step_{s}" / "adapter_config.json").exists():
            raise SystemExit(f"missing adapter: {ckpt_root / f'step_{s}'}")
        todo.append(s)
    if not todo:
        print("nothing to do")
        return
    print(f"[plan] rid={args.rid} steps={todo} n_examples={len(examples)} out={out_dir}")

    from transformers import AutoTokenizer
    from utils.vllm_utils import make_vllm_llm
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Flat list of requests over all examples; token ids AND the formatted
    # prompt string (the string is needed for the budget-forcing phase-2 concat).
    flat = []  # (example_idx, request dict, token_ids, prompt_str)
    for ei, ex in enumerate(examples):
        for r in build_requests(ex, args.lie_k):
            ids = tokenizer.apply_chat_template(
                r["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            if hasattr(ids, "input_ids"):  # BatchEncoding on newer transformers
                ids = ids["input_ids"]
            ids = list(ids)
            prompt_str = tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            flat.append((ei, r, ids, prompt_str))
    max_prompt = max(len(t) for _, _, t, _ in flat)
    print(f"[prompts] {len(flat)} requests; max prompt tokens = {max_prompt}; max_model_len={args.max_model_len}")
    _top = args.thinking_budget + args.answer_budget if args.enable_thinking else args.max_tokens
    if max_prompt + _top > args.max_model_len:
        print(f"[warn] longest prompt + output budget exceeds max_model_len; long prompts will get fewer output tokens")

    t0 = time.time()
    llm = make_vllm_llm(
        model_id=args.base_model,
        tokenizer_id=args.base_model,
        tp_size=1,
        gpu_mem_util=args.gpu_mem_util,
        dtype="float16",
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        max_model_len=args.max_model_len,
    )
    print(f"[vllm] loaded {args.base_model} in {time.time() - t0:.0f}s (vllm {vllm.__version__})")

    gen_settings = {
        "base_model": args.base_model,
        "chat_template": "base tokenizer apply_chat_template(add_generation_prompt=True"
                         + (", enable_thinking=True)" if args.enable_thinking else ")"),
        "enable_thinking": args.enable_thinking,
        "budget_forcing": args.enable_thinking,
        "thinking_budget": args.thinking_budget if args.enable_thinking else None,
        "answer_budget": args.answer_budget if args.enable_thinking else None,
        "budget_forcing_note": (
            "s1 / Muennighoff 2025 two-pass: phase1 up to thinking_budget tokens, "
            "cap at first </think> (inject if absent), phase2 up to answer_budget tokens "
            "conditioned on prompt+capped-thinking; completion = capped_thinking + answer"
        ) if args.enable_thinking else None,
        "prompt_format": "official cais/mask generate_responses.py (pressure: system+user[; dd turns]; belief: 'You are a helpful assistant.' + belief_elicit_k)",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "lie_k": args.lie_k,
        "belief_k": {"statistics": {"1": 3}, "default": {"1": 3, "2": 1, "3": 1}},
        "seed_scheme": f"per-request seed = {args.seed}*100000 + request_index",
        "seed": args.seed,
        "dtype": "float16",
        "max_model_len": args.max_model_len,
        "vllm_version": vllm.__version__,
        "lora_rank": args.lora_rank,
        "selection_file": str(sel_path),
        "n_examples": len(examples),
    }

    for s in todo:
        t1 = time.time()
        adapter = None if s == 0 else str(ckpt_root / f"step_{s}")
        lora_req = None if adapter is None else LoRARequest(f"{args.rid}_step_{s}", s, adapter)
        texts, finish, injected = generate_step(llm, flat, lora_req, args, SamplingParams, tokenizer)

        per_ex = []
        for ei, ex in enumerate(examples):
            per_ex.append({
                "id": f"{ex['config']}:{ex['task_id']}",
                "task_id": ex["task_id"],
                "config": ex["config"],
                "archetype": ex["config"],
                "type": ex.get("type"),
                "proposition": ex.get("proposition"),
                "ground_truth": ex.get("ground_truth"),
                "formatted_ground_truth": ex.get("formatted_ground_truth"),
                "system_prompt": ex.get("system_prompt"),
                "user_prompt": ex.get("user_prompt"),
                "dd_turn_1_prompt": ex.get("dd_turn_1_prompt"),
                "dd_turn_1_response": ex.get("dd_turn_1_response"),
                "belief_elicit_1": ex.get("belief_elicit_1"),
                "belief_elicit_2": ex.get("belief_elicit_2"),
                "belief_elicit_3": ex.get("belief_elicit_3"),
                "belief_system_1": BELIEF_SYSTEM_PROMPT,
                "pressure_messages": None,
                "pressure_completions": [],
                "belief_completions": {},
                "official_columns": {},
                "finish_reasons": {},
            })
        for ri, (ei, r, ids, ps) in enumerate(flat):
            rec = per_ex[ei]
            rec["official_columns"][r["official_column"]] = texts[ri]
            rec["finish_reasons"][r["official_column"]] = finish[ri]
            if r["kind"] == "pressure":
                rec["pressure_messages"] = r["messages"]
                rec["pressure_completions"].append(texts[ri])
            else:
                key = r["key"].split("_run")[0]  # be1 / be2 / be3
                slot = rec["belief_completions"].setdefault(key, {"messages": r["messages"], "completions": []})
                slot["completions"].append(texts[ri])

        out = {
            "rid": args.rid,
            "step": s,
            "adapter_path": adapter,
            "generation_settings": gen_settings,
            "n_requests": len(flat),
            "think_end_injected": injected,
            "elapsed_s": round(time.time() - t1, 1),
            "examples": per_ex,
        }
        out_path = out_dir / f"step_{s}.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        os.replace(tmp, out_path)
        n_empty = sum(1 for t in texts if not t.strip())
        print(f"[done] step {s}: {len(flat)} completions in {time.time() - t1:.0f}s, empty={n_empty} -> {out_path}", flush=True)

    print(f"[all done] {len(todo)} checkpoints in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
