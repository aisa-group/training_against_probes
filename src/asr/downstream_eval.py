"""Shared checkpoint evaluation: held-out StrongREJECT and utility aggregation.

Used by inline training evaluation and standalone checkpoint/baseline evaluators.
Utility measurements themselves are supplied by the utility evaluation pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# filelock workaround for /fast (no flock support)
import filelock
filelock.FileLock = filelock.SoftFileLock
filelock.UnixFileLock = filelock.SoftFileLock

_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "probes"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

from utils import paths as _paths  # noqa: E402
from asr.judges import score_strongreject_batch  # noqa: E402
from utils.vllm_utils import generate_vllm_subprocess  # noqa: E402

LORA_ROOT = _paths.LORA_ROOT
RESULTS_ROOT = _paths.LIVE_DOWNSTREAM


def aggregate_utility(utility_results: dict) -> dict:
    """Compute the aggregate utility score.

    aggregate = (mmlu + gsm8k + (ifeval_prompt + ifeval_instr)/2) / 3
    Returns dict with components + aggregate.
    """
    mmlu = float(utility_results.get("utility/mmlu_accuracy", float("nan")))
    gsm8k = float(utility_results.get("utility/gsm8k_accuracy", float("nan")))
    ifeval_p = float(utility_results.get("utility/ifeval_prompt_accuracy", float("nan")))
    ifeval_i = float(utility_results.get("utility/ifeval_instruction_accuracy", float("nan")))
    ifeval_avg = (ifeval_p + ifeval_i) / 2.0
    aggregate = (mmlu + gsm8k + ifeval_avg) / 3.0
    return {
        "mmlu": mmlu,
        "gsm8k": gsm8k,
        "ifeval_prompt": ifeval_p,
        "ifeval_instr": ifeval_i,
        "ifeval_avg": ifeval_avg,
        "aggregate": aggregate,
    }


def _answer_only(text: str) -> str:
    """For a reasoning model, StrongREJECT should score the ANSWER, not the
    thinking trace: the detector is trained on the answer tokens, so the metric
    must measure the same thing. Returns the text after the last </think>; a
    no-op for completions without a thinking block (Mistral, Llama)."""
    marker = "</think>"
    return text.rsplit(marker, 1)[-1].strip() if marker in text else text


def run_strongreject_for_step(
    *,
    run_id: str,
    step: int,
    base_model: str,
    base_tokenizer: str,
    jbb_prompts: list[str],
    jbb_formatted: list[str],
    out_dir: Path,
    max_new_tokens: int = 256,
) -> dict:
    """Generate JBB completions with the step's LoRA adapter and score with
    StrongREJECT. Returns dict with `mean_p_harmful` etc. vLLM and judge each
    run in their own subprocesses (memory freed between calls)."""
    lora_path = str(LORA_ROOT / run_id / f"step_{step}")

    print(f"[step {step}] StrongReject: generating {len(jbb_formatted)} completions via vLLM...")
    completions = generate_vllm_subprocess(
        formatted_prompts=jbb_formatted,
        model_name=base_model,
        tokenizer_name=base_tokenizer,
        lora_path=lora_path,
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=0.85,
        lora_rank=64,
    )

    print(f"[step {step}] StrongReject: scoring with judge subprocess...")
    answers = [_answer_only(c) for c in completions]
    _stripped = sum(1 for c, a in zip(completions, answers) if a != c)
    if _stripped:
        print(f"[step {step}] scoring ANSWER only for {_stripped}/{len(completions)} "
              f"reasoning completions (post-</think>)")
    scores = score_strongreject_batch(jbb_prompts, answers)
    mean = float(np.mean(scores))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{step}.json"
    out_path.write_text(json.dumps({
        "step": step,
        "judge": "strongreject",
        "n_prompts": len(jbb_prompts),
        "mean_p_harmful": mean,
        "scored_region": "answer_only" if any(a != c for a, c in zip(answers, completions)) else "full",
        "results": [
            {"prompt": p, "completion": c, "answer": a, "score": s}
            for p, c, a, s in zip(jbb_prompts, completions, answers, scores)
        ],
    }, indent=2))

    print(f"[step {step}] StrongReject mean p_harmful = {mean:.4f}")
    return {"mean": mean, "mean_p_harmful": mean, "n_prompts": len(jbb_prompts)}
