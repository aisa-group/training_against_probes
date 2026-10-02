"""Shared experiment paths.

NM_ROOT selects the output root for checkpoints, evaluation results, and logs.
Dataset and steering inputs have separate paths and are not relocated by NM_ROOT.
"""

import os
from pathlib import Path

NM_ROOT = Path(os.environ.get("NM_ROOT", "/data/new_master"))

# Per-run LoRA checkpoints + probe snapshots: <LORA_ROOT>/<run_id>/step_<N>/
LORA_ROOT = NM_ROOT / "lora-finetuned"

# downstream_eval.py output: baseline_utility.json, utility/, strongreject/
LIVE_DOWNSTREAM = NM_ROOT / "live_downstream"

# Offline eval outputs
EVAL_GENERATIONS = NM_ROOT / "eval_generations"
UTILITY = NM_ROOT / "utility"

# Scratch dir used to hand the current adapter to the vLLM subprocess
FINETUNE_SAVE = NM_ROOT / "finetune"

# wandb.init(dir=...)
WANDB_DIR = NM_ROOT / "wandb"

# Shared inputs (KL anchor files, MASK selection, paired probe data).
# Defaults to <NM_ROOT>/datasets; set DATASETS_DIR to keep them elsewhere.
DATASETS = Path(os.environ.get("DATASETS_DIR", NM_ROOT / "datasets"))


def describe() -> str:
    """One-line summary for startup logs, so a run records which tree it wrote to."""
    return f"NM_ROOT={NM_ROOT} (set via env)" if "NM_ROOT" in os.environ else f"NM_ROOT={NM_ROOT} (default)"
