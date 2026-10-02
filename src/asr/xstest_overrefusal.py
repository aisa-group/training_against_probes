"""
Over-refusal experiment on XSTest.

For each checkpoint we generate on the 250 XSTest *safe* prompts (prompts that
LOOK unsafe but are benign) and score every response with the StrongReject
judge. StrongReject returns 0 = refused ... 1 = fully compliant, so on safe
prompts a LOW mean StrongReject == more over-refusal (the model wrongly refuses
benign requests). We report mean +/- sample std (ddof=1) per checkpoint.

Checkpoints (each finetuned run taken at its last >=80%-utility step):
  Llama (abliterated heretic base):
    base (no LoRA), Frozen RUN_026@100, Cont.updated RUN_027@15, Retrained RUN_025@16
  Mistral 7B Instruct v0.1:
    base (no LoRA), Frozen RUN_022@30, Cont.updated RUN_028@9, Retrained RUN_010@12

Output: /data/new_master/xstest_overrefusal/{model}_{label}.json + summary.json

Usage (cluster, 1 GPU):
  python -m asr.xstest_overrefusal
  python -m asr.xstest_overrefusal --only llama --max_new_tokens 64 --n_prompts 8   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOFTFILELOCK", "1")
import filelock  # noqa: E402
filelock.FileLock = filelock.SoftFileLock
filelock.UnixFileLock = filelock.SoftFileLock

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from utils.finetune_utils import apply_chat_template  # noqa: E402
from utils.vllm_utils import generate_vllm_subprocess  # noqa: E402
from asr.judges import score_strongreject_batch  # noqa: E402

LORA_ROOT = Path("/data/new_master/lora-finetuned")
OUT_DIR = Path("/data/new_master/xstest_overrefusal")

# base model + tokenizer per model family
MODELS = {
    "llama": {
        "model_id": os.environ.get("ABLITERATED_MODEL", "lenalibon/Meta-Llama-3-8B-Instruct-heretic-mlabonne"),
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "display": "Abliterated Llama 3 8B Instruct",
        "checkpoints": [
            ("Abliterated (base)", None, None),
            ("Frozen, 80% utility",        "RUN_026", 100),
            ("Continuously updated, 80% utility", "RUN_027", 15),
            ("Retrained, 80% utility",     "RUN_025", 16),
        ],
    },
    "llama_instruct": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "display": "Llama 3 8B Instruct (non-abliterated)",
        "checkpoints": [
            ("Llama 3 8B Instruct (non-abliterated)", None, None),
        ],
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "tokenizer_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "display": "Mistral 7B Instruct v0.1",
        "checkpoints": [
            ("Mistral 7B Instruct v0.1 (base)", None, None),
            ("Frozen, 80% utility",        "RUN_022", 30),
            ("Continuously updated, 80% utility", "RUN_028", 9),
            ("Retrained, 80% utility",     "RUN_010", 12),
        ],
    },
}


def load_xstest_safe(n_prompts: int | None) -> list[str]:
    ds = load_dataset("walledai/XSTest")["test"]
    safe = [r["prompt"] for r in ds if r["label"] == "safe"]
    if n_prompts is not None:
        safe = safe[:n_prompts]
    return safe


def eval_checkpoint(label, run_id, step, model_id, tokenizer_id, tokenizer,
                    formatted, prompts, max_new_tokens, batch_hint=None) -> dict:
    lora_path = str(LORA_ROOT / run_id / f"step_{step}") if run_id else None
    print(f"\n[{label}] lora={lora_path} — generating {len(formatted)} XSTest-safe completions ...",
          flush=True)
    completions = generate_vllm_subprocess(
        formatted_prompts=formatted,
        model_name=model_id, tokenizer_name=tokenizer_id,
        lora_path=lora_path, max_new_tokens=max_new_tokens,
        gpu_memory_utilization=0.85, lora_rank=64,
    )
    print(f"[{label}] scoring with StrongReject (0=refused, 1=compliant) ...", flush=True)
    scores = score_strongreject_batch(prompts, completions)
    arr = np.asarray(scores, dtype=np.float64)
    rec = {
        "label": label, "run_id": run_id, "step": step,
        "model_id": model_id, "dataset": "XSTest-safe(250)",
        "metric": "mean_strongreject_on_safe (higher=more compliant, lower=more over-refusal)",
        "n": int(arr.size),
        "mean_strongreject": float(arr.mean()),
        "std_strongreject": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "results": [
            {"prompt": p, "completion": c, "score": float(s)}
            for p, c, s in zip(prompts, completions, scores)
        ],
    }
    print(f"[{label}] mean StrongReject on safe = {rec['mean_strongreject']:.4f} "
          f"+/- {rec['std_strongreject']:.4f}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="llama or mistral")
    ap.add_argument("--n_prompts", type=int, default=None, help="cap safe prompts (smoke)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_xstest_safe(args.n_prompts)
    print(f"[xstest] {len(prompts)} safe prompts")

    keys = [args.only] if args.only else list(MODELS)
    summary = []
    for key in keys:
        cfg = MODELS[key]
        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        formatted = apply_chat_template(prompts, tokenizer)
        for label, run_id, step in cfg["checkpoints"]:
            rec = eval_checkpoint(
                label, run_id, step, cfg["model_id"], cfg["tokenizer_id"],
                tokenizer, formatted, prompts, args.max_new_tokens,
            )
            fname = f"{key}_{(run_id or 'base')}.json"
            (out_dir / fname).write_text(json.dumps(rec, indent=2))
            summary.append({
                "model": key, "display": cfg["display"],
                "label": label, "run_id": run_id, "step": step,
                "n": rec["n"], "mean_strongreject": rec["mean_strongreject"],
                "std_strongreject": rec["std_strongreject"],
            })
            (out_dir / "summary.json").write_text(json.dumps({"rows": summary}, indent=2))

    print("\n| Model | Checkpoint | mean StrongReject (safe) | std | n |")
    print("|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['model']} | {s['label']} | {s['mean_strongreject']:.4f} | "
              f"{s['std_strongreject']:.4f} | {s['n']} |")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
