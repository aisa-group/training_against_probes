"""Evaluate the base model on JailbreakBench with StrongREJECT.

Uses the same prompts, greedy decoding, and judge as checkpoint evaluation.
Writes {ROOT}/lmeval_rescore/_baseline_sr_{slug}.json."""
import json
import os
import re
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (SRC, os.path.join(SRC, "asr"), os.path.join(SRC, "utils")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402

from utils.jailbreak_datasets import load_jailbreakbench  # noqa: E402
from utils.finetune_utils import apply_chat_template  # noqa: E402
from utils.vllm_utils import generate_vllm_subprocess  # noqa: E402
from asr.judges import score_strongreject_batch  # noqa: E402

ROOT = os.environ.get("NM_ROOT", "/data/new_master")
OUT = os.path.join(ROOT, "lmeval_rescore")
MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
# Use the original tokenizer when converted model metadata is incompatible with vLLM.
TOKENIZER = os.environ.get("BASE_TOKENIZER") or MODEL
MAX_NEW_TOKENS = 256          # identical to run_strongreject_for_step


def main():
    os.makedirs(OUT, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", MODEL).strip("_")
    path = f"{OUT}/_baseline_sr_{slug}.json"
    if os.path.exists(path):
        print(f"[base-sr] {path} exists, nothing to do")
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    prompts = load_jailbreakbench()
    formatted = apply_chat_template(prompts, tok)
    print(f"[base-sr] {len(prompts)} JBB prompts, no adapter", flush=True)

    completions = generate_vllm_subprocess(
        formatted_prompts=formatted, model_name=MODEL, tokenizer_name=TOKENIZER,
        lora_path=None, max_new_tokens=MAX_NEW_TOKENS,
        gpu_memory_utilization=0.85, lora_rank=64,
    )
    scores = score_strongreject_batch(prompts, completions)
    mean = float(np.mean(scores))
    json.dump({"base_model": MODEL, "tokenizer": TOKENIZER,
               "judge": "strongreject",
               "n_prompts": len(prompts), "mean_p_harmful": mean,
               "results": [{"prompt": p, "completion": c, "score": s}
                           for p, c, s in zip(prompts, completions, scores)]},
              open(path, "w"), indent=2)
    print(f"[base-sr] base-model StrongREJECT = {mean:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
