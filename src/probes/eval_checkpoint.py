"""Evaluate saved LoRA checkpoints with utility metrics and StrongREJECT.

Uses utils.lmeval_utils and asr.downstream_eval for the same metrics as training.
Results are written to live_downstream/<run_id>/{utility,strongreject}/step_<N>.json.
Base utility is cached in lora-finetuned/<run_id>/baseline_utility.json.
Base StrongREJECT is computed separately by baseline_strongreject.py.

StrongREJECT subprocesses finish before the utility engine is loaded, avoiding
simultaneous GPU reservations. The vLLM utility engine is reused across adapters.

Usage:
    RUN_ID=... BASE_MODEL=... python src/probes/eval_checkpoint.py
    RUN_ID=... BASE_MODEL=... STEPS=0,5,10 python src/probes/eval_checkpoint.py"""
import gc
import json
import os
import sys
from pathlib import Path

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SRC, os.path.join(SRC, "utils"), os.path.join(SRC, "asr")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROOT = Path(os.environ.get("NM_ROOT", "/data/new_master"))
LORA_ROOT = ROOT / "lora-finetuned"
LIVE = ROOT / "live_downstream"

RUN_ID = os.environ["RUN_ID"]
BASE_MODEL = os.environ["BASE_MODEL"]
BASE_TOKENIZER = os.environ.get("BASE_TOKENIZER") or BASE_MODEL
# Empty/absent STEPS means "every checkpoint that exists, plus step 0".
_STEPS = os.environ.get("STEPS", "").strip()
THRESHOLD_PCT = float(os.environ.get("UTILITY_THRESHOLD_PCT", "0.8"))
SR_N = int(os.environ.get("SR_N_PROMPTS", "100"))


def _discover_steps():
    d = LORA_ROOT / RUN_ID
    found = sorted(int(p.name.split("_")[1]) for p in d.glob("step_*")
                   if p.name.split("_")[1].isdigit()
                   and (p / "adapter_config.json").exists())
    return [0] + found


def main():
    import torch
    from asr.downstream_eval import aggregate_utility, run_strongreject_for_step
    from utils.lmeval_utils import (build_lm, build_lm_hf, evaluate, vllm_supports,
                                    EVAL_ENV)
    from utils.finetune_utils import apply_chat_template
    from transformers import AutoTokenizer
    from utils.jailbreak_datasets import load_jailbreakbench

    steps = ([int(x) for x in _STEPS.split(",") if x.strip()] if _STEPS
             else _discover_steps())
    if not steps:
        raise SystemExit(f"no checkpoints under {LORA_ROOT / RUN_ID}")
    print(f"[eval] {RUN_ID}: {len(steps)} checkpoints {steps[:3]}...{steps[-1:]}",
          flush=True)

    def adapter_for(step):
        if step == 0:
            return None
        p = LORA_ROOT / RUN_ID / f"step_{step}"
        if not p.exists():
            raise SystemExit(f"no adapter at {p}")
        return str(p)

    os.environ.update(EVAL_ENV)
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    jbb = load_jailbreakbench()[:SR_N]
    jbb_fmt = apply_chat_template(jbb, tok)

    base_path = LORA_ROOT / RUN_ID / "baseline_utility.json"
    baseline = json.loads(base_path.read_text()) if base_path.exists() else None

    # Complete StrongREJECT before loading the persistent utility engine to avoid GPU contention.
    for step in steps:
        # Skip base-model safety scoring here; baseline_strongreject.py computes it without a LoRA adapter.
        if step == 0:
            continue
        sr_path = LIVE / RUN_ID / "strongreject" / f"step_{step}.json"
        if sr_path.exists():
            print(f"[eval] step {step}: StrongREJECT already present", flush=True)
            continue
        run_strongreject_for_step(
            run_id=RUN_ID, step=step, base_model=BASE_MODEL,
            base_tokenizer=BASE_TOKENIZER, jbb_prompts=jbb,
            jbb_formatted=jbb_fmt, out_dir=LIVE / RUN_ID / "strongreject",
        )
    gc.collect()
    torch.cuda.empty_cache()
    print("[eval] StrongREJECT pass done; building the utility engine", flush=True)

    # Load the utility engine after the StrongREJECT subprocesses exit.
    any_adapter = next((adapter_for(s) for s in steps if s > 0), None)
    # UTIL_GPU_MEM controls the utility engine reservation; reduce it if generation exceeds GPU memory.
    _util_gpu_mem = float(os.environ.get("UTIL_GPU_MEM", "0.85"))
    if vllm_supports(BASE_MODEL):
        lm = build_lm(BASE_MODEL, BASE_TOKENIZER, any_adapter=any_adapter,
                      gpu_memory_utilization=_util_gpu_mem)
    else:
        lm = None       # HF backend cannot swap adapters; built per step below

    for step in steps:
        up = LIVE / RUN_ID / "utility" / f"step_{step}.json"
        if up.exists():
            print(f"[eval] step {step}: utility already present", flush=True)
            continue
        ad = adapter_for(step)
        raw = (evaluate(lm, ad) if lm is not None
               else evaluate(build_lm_hf(BASE_MODEL, BASE_TOKENIZER, ad), ad))
        # evaluate() returns the utility-prefixed keys expected by aggregate_utility.
        cur = aggregate_utility(raw)

        if step == 0 or baseline is None:
            baseline = cur
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(json.dumps(cur, indent=2))

        thr = THRESHOLD_PCT * baseline["aggregate"]
        below = cur["aggregate"] < thr
        up.parent.mkdir(parents=True, exist_ok=True)
        up.write_text(json.dumps({
            "step": step, "baseline": baseline, "threshold": thr, "current": cur,
            "below_threshold": bool(below),
            "scorer": "lm_eval (scored in-loop)",
        }, indent=2))
        print(f"[eval] step {step}: aggregate={cur['aggregate']:.4f} "
              f"(threshold {thr:.4f})" + ("  << BELOW" if below else ""), flush=True)


if __name__ == "__main__":
    main()
