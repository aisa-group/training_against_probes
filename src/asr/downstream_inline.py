"""Run held-out JailbreakBench StrongREJECT evaluation during finetuning.

Generation and scoring use the shared checkpoint evaluator in downstream_eval.
MMLU/GSM8K/IFEval utility is evaluated separately by finetune_eval.run_eval.
"""
from __future__ import annotations

import os

import wandb  # noqa: E402

from asr.downstream_eval import (  # noqa: E402
    RESULTS_ROOT,
    run_strongreject_for_step,
)
from utils.jailbreak_datasets import load_jailbreakbench  # noqa: E402
from utils.finetune_utils import apply_chat_template  # noqa: E402


class _AllowHubAccess:
  

    _VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")

    def __enter__(self):
        self._prev = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ[k] = "0"
        return self

    def __exit__(self, *exc):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class InlineDownstream:
    """Held-out StrongREJECT for one step at a time."""

    def __init__(self, tokenizer, run_id, base_model, base_tokenizer,
                 max_new_tokens=256):
        self.run_id = run_id
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.max_new_tokens = max_new_tokens

        self.out_strongreject = RESULTS_ROOT / run_id / "strongreject"

        self.jbb_prompts = load_jailbreakbench()
        self.jbb_formatted = apply_chat_template(self.jbb_prompts, tokenizer)

    def run_step(self, step: int) -> dict:
        """Held-out eval for ``step``. Metric names match the watcher's exactly."""
        with _AllowHubAccess():
            sr = run_strongreject_for_step(
                run_id=self.run_id, step=step,
                base_model=self.base_model, base_tokenizer=self.base_tokenizer,
                jbb_prompts=self.jbb_prompts, jbb_formatted=self.jbb_formatted,
                out_dir=self.out_strongreject, max_new_tokens=self.max_new_tokens,
            )

        log_dict = {
            "asr_step": step,
            "jailbreak/strongreject_harmfulscore": sr["mean"],
        }
        wandb.log(log_dict)
        print(f"  [downstream] step {step}: held-out StrongREJECT={sr['mean']:.4f}",
              flush=True)
        return log_dict
