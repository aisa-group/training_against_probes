"""
ClearHarm generation eval for a finetuned LoRA run.
"""

import argparse
import json
import os
import sys

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SRC_DIR)

from datasets import load_dataset
from transformers import AutoTokenizer

from utils.finetune_utils import apply_chat_template, generate_on_policy_vllm

from utils import paths as _paths

LORA_ROOT = str(_paths.LORA_ROOT)
SAVE_DIR = str(_paths.NM_ROOT / "generations_clearharm")
MAX_NEW_TOKENS = 512
GPU_MEMORY_UTILIZATION = 0.85

_HERETIC_TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct"


def _read_adapter_config(adapter_path: str) -> dict:
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        return json.load(f)


def _model_and_tokenizer(adapter_cfg: dict) -> tuple[str, str]:
    base_model = adapter_cfg["base_model_name_or_path"]
    if "heretic" in base_model.lower():
        return base_model, _HERETIC_TOKENIZER
    return base_model, base_model


def _all_adapter_dirs(run_id: str) -> list[tuple[str, str]]:
    """Return (label, path) pairs for every checkpoint in the run, sorted by step."""
    run_dir = os.path.join(LORA_ROOT, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    entries = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            continue
        entries.append((name, path))

    def _sort_key(item):
        name = item[0]
        if name.startswith("step_"):
            return (0, int(name.split("_")[1]))
        return (1, 0)  # final_lora last

    return sorted(entries, key=_sort_key)


def _generate_for_adapter(
    adapter_path: str,
    prompts: list[str],
    tokenizer,
    model_name: str,
    tokenizer_name: str,
    lora_rank: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
) -> list[str]:
    formatted = apply_chat_template(prompts, tokenizer)
    return generate_on_policy_vllm(
        formatted_prompts=formatted,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        lora_path=adapter_path,
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_rank=lora_rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="W&B run ID")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--gpu_memory_utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    args = parser.parse_args()

    checkpoints = _all_adapter_dirs(args.run_id)
    print(f"Run {args.run_id}: {len(checkpoints)} checkpoints found")

    # All checkpoints share the same base model — read from the first one.
    _, first_path = checkpoints[0]
    adapter_cfg = _read_adapter_config(first_path)
    model_name, tokenizer_name = _model_and_tokenizer(adapter_cfg)
    lora_rank = int(adapter_cfg.get("r", 64))
    print(f"Base model : {model_name}")
    print(f"Tokenizer  : {tokenizer_name}")
    print(f"LoRA rank  : {lora_rank}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading AlignmentResearch/ClearHarm ...")
    ds = load_dataset("AlignmentResearch/ClearHarm", split="train")
    prompts = [row["content"][0] for row in ds]
    print(f"  {len(prompts)} prompts")

    out_dir = os.path.join(SAVE_DIR, args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    for label, adapter_path in checkpoints:
        out_path = os.path.join(out_dir, f"{label}.json")
        if os.path.isfile(out_path):
            print(f"[skip] {label} — already done ({out_path})")
            continue

        print(f"\n[{label}] generating {len(prompts)} completions ...")
        for attempt in range(1, 4):
            try:
                completions = _generate_for_adapter(
                    adapter_path=adapter_path,
                    prompts=prompts,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    tokenizer_name=tokenizer_name,
                    lora_rank=lora_rank,
                    max_new_tokens=args.max_new_tokens,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                break
            except RuntimeError as e:
                print(f"[{label}] attempt {attempt}/3 failed: {e}")
                if attempt == 3:
                    print(f"[{label}] skipping after 3 failed attempts")
                    completions = None
        if completions is None:
            continue

        records = [
            {
                "idx": i,
                "prompt": prompts[i],
                "completion": completions[i],
            }
            for i in range(len(prompts))
        ]
        payload = {
            "run_id": args.run_id,
            "adapter_path": adapter_path,
            "model_name": model_name,
            "n_samples": len(records),
            "samples": records,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{label}] saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
