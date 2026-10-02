"""Generate fixed base-model completions for KL regularization.

SOURCE selects math, ultrachat, or dolci. Generate the reference text once and
reuse it throughout finetuning. TAG distinguishes outputs from different models.
The KL objective compares base and adapted distributions on this fixed text.

Math outputs include a shuffled prompt pool, all rollouts, and a subset with
correct final answers. Correctness uses lm-evaluation-harness's MATH checker.
Conversational sources produce all rollouts without a correctness filter.
Prompts match the text used during training; no extra answer-format instruction
is added. Outputs are written under {ROOT}/datasets/."""
import json
import os
import random
import re
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SRC, os.path.join(SRC, "utils")):
    sys.path.insert(0, _p)

ROOT = os.environ.get("NM_ROOT", "/data/new_master")
OUT = os.path.join(ROOT, "datasets")
MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "1000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
SOURCE = os.environ.get("SOURCE", "math")
# Use the original tokenizer if the converted model retains its vocabulary.
TOKENIZER = os.environ.get("TOKENIZER") or MODEL
# Use TAG to separate reference text produced by different base models.
TAG = os.environ.get("ROLLOUT_TAG", "")
_SUF = f"_{TAG}" if TAG else ""
SEED = 42

# Only math sources provide gold answers for correctness filtering.
SCORED = SOURCE == "math"

SUBJECTS = ["algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"]


def build_pool():
    """(prompt, gold solution or None, subject tag), shuffled, N_PROMPTS kept."""
    from datasets import load_dataset

    rows = []
    if SOURCE == "math":
        for subj in SUBJECTS:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            for ex in ds:
                q, sol = ex.get("problem"), ex.get("solution")
                if q and sol:
                    rows.append({"prompt": q, "solution": sol, "subject": subj})
    else:
        # Use the training loaders to select the same prompt pool.
        from utils.finetune_utils import load_kl_dataset
        name = {"ultrachat": "HuggingFaceH4/ultrachat_200k",
                "dolci": f"{ROOT}/datasets/dolci_prompts.json"}[SOURCE]
        # Load a larger pool before shuffling to avoid source-order bias.
        for q in load_kl_dataset(name, max_n=max(N_PROMPTS * 5, 5000)):
            rows.append({"prompt": q, "solution": None, "subject": SOURCE})
    random.Random(SEED).shuffle(rows)
    return rows[:N_PROMPTS]


def is_degenerate(text, n=4, k=5, min_words=40):
    """Detect repetitive generations when preparing the KL reference pool."""
    w = text.split()
    if len(w) < min_words:
        return False
    counts = {}
    for i in range(len(w) - n + 1):
        g = " ".join(w[i:i + n])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= k:
            return True
    return False


def load_checker():
    """lm_eval's Hendrycks MATH answer checker. Pure python, no sympy."""
    from lm_eval.tasks.hendrycks_math.utils import (  # noqa: F401
        is_equiv, last_boxed_only_string, remove_boxed,
    )
    return is_equiv, last_boxed_only_string, remove_boxed


def extract_pred(text, last_boxed_only_string, remove_boxed):
    """The model's final answer, if one can be recovered.

    Prefer \\boxed{}, which is what the checker's normalisers expect. Fall back
    to the last number in the text, which is what a model that ignored the LaTeX
    convention will have produced. Returns None when neither is present, and
    those rollouts count as not-correct rather than being silently dropped.
    """
    boxed = last_boxed_only_string(text)
    if boxed:
        try:
            return remove_boxed(boxed)
        except Exception:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _generate_hf(formatted_prompts, tok, max_new_tokens, batch_size=None):
    """Greedy batched generation with transformers, for models vLLM cannot serve.

    The prompts arrive already chat-formatted, so they are tokenised as raw text
    and NOT re-templated -- double-templating a Qwen3 prompt would insert a
    second <|im_start|> block and change what the base model is conditioned on.

    Left padding: decoder-only batched generation right-pads by default, which
    would start every continuation after a run of pad tokens.
    """
    import torch
    from transformers import AutoModelForCausalLM

    bs = batch_size or int(os.environ.get("HF_GEN_BATCH", "8"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    out = []
    for i in range(0, len(formatted_prompts), bs):
        chunk = formatted_prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(ids[j][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
        if (i // bs) % 5 == 0:
            print(f"[build]   generated {len(out)}/{len(formatted_prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    is_equiv = last_boxed = remove_boxed = None
    if SCORED:
        is_equiv, last_boxed, remove_boxed = load_checker()
    print(f"[build] source={SOURCE}  n={N_PROMPTS}  scored={SCORED}", flush=True)

    pool = build_pool()
    subj_counts = {}
    for r in pool:
        subj_counts[r["subject"]] = subj_counts.get(r["subject"], 0) + 1
    print(f"[build] pool: {len(pool)} problems across {len(subj_counts)} subjects")
    for s in SUBJECTS:
        print(f"           {s:26s} {subj_counts.get(s, 0)}")

    p_path = os.path.join(OUT, f"{SOURCE}_prompts_shuffled.json")
    json.dump([r["prompt"] for r in pool], open(p_path, "w"))
    print(f"[build] wrote {p_path}", flush=True)

    from transformers import AutoTokenizer
    from utils.finetune_utils import apply_chat_template, set_enable_thinking
    # Include reasoning traces when the training model generates them.
    set_enable_thinking(os.environ.get("ENABLE_THINKING", "0").lower()
                        in ("1", "true", "yes"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    formatted = apply_chat_template([r["prompt"] for r in pool], tok)
    print(f"[build] generating {len(formatted)} solutions from the BASE model "
          f"(no adapter), greedy, max_new_tokens={MAX_NEW_TOKENS}", flush=True)

    # Use transformers for Qwen3, which is unsupported by the pinned vLLM version.
    from utils.lmeval_utils import vllm_supports
    if vllm_supports(MODEL):
        from utils.vllm_utils import generate_vllm_subprocess
        completions = generate_vllm_subprocess(
            formatted_prompts=formatted, model_name=MODEL, tokenizer_name=TOKENIZER,
            lora_path=None, max_new_tokens=MAX_NEW_TOKENS,
            gpu_memory_utilization=0.85, lora_rank=64,
        )
    else:
        print(f"[build] {MODEL} is not a vLLM 0.7.3 architecture; "
              f"generating with transformers", flush=True)
        completions = _generate_hf(formatted, tok, MAX_NEW_TOKENS)

    all_rows, ok_rows = [], []
    n_extract = n_degen = 0
    for r, c in zip(pool, completions):
        c = c or ""
        gold = pred = None
        correct = False
        if SCORED:
            gold_boxed = last_boxed(r["solution"])
            if gold_boxed:
                try:
                    gold = remove_boxed(gold_boxed)
                except Exception:
                    gold = None
            pred = extract_pred(c, last_boxed, remove_boxed)
            n_extract += pred is not None
            if pred is not None and gold is not None:
                try:
                    correct = bool(is_equiv(pred, gold))
                except Exception:
                    correct = False
        degen = is_degenerate(c)
        n_degen += degen
        row = {"prompt": r["prompt"], "completion": c, "subject": r["subject"],
               "gold": gold, "pred": pred, "correct": correct,
               "degenerate": degen}
        all_rows.append(row)
        if correct and not degen:
            ok_rows.append(row)

    a_path = os.path.join(OUT, f"{SOURCE}{_SUF}_kl_rollouts_all.json")
    json.dump(all_rows, open(a_path, "w"))
    o_path = None
    if SCORED:
        o_path = os.path.join(OUT, f"{SOURCE}{_SUF}_kl_rollouts_correct.json")
        json.dump(ok_rows, open(o_path, "w"))

    n = len(all_rows)
    print()
    print(f"[build] rollouts        : {n}")
    if SCORED:
        print(f"[build] answer extracted: {n_extract} ({n_extract / n:.1%})")
        print(f"[build] correct         : {len(ok_rows)} ({len(ok_rows) / n:.1%})")
    print(f"[build] degenerate      : {n_degen} ({n_degen / n:.1%})")
    print(f"[build] wrote {a_path}")
    if o_path:
        print(f"[build] wrote {o_path}")
    print()
    print("=" * 78)
    print("THREE ROLLOUTS IN FULL (read these before launching anything)")
    print("=" * 78)
    for row in all_rows[:3]:
        print(f"\n--- subject={row['subject']}  correct={row['correct']}  "
              f"gold={row['gold']!r}  pred={row['pred']!r}")
        print(f"PROBLEM: {row['prompt'][:300]}")
        print(f"SOLUTION:\n{row['completion'][:1200]}")


if __name__ == "__main__":
    main()
