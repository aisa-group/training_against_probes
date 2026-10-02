"""
Additional MMLU / GSM8K / IFEval utility evals for a finetuned LoRA checkpoint.

"""

import json
import os
import sys

# ── make the rest of src/ importable when run as a script ───────────────────
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_SRC_DIR, "probes"))
sys.path.insert(0, _SRC_DIR)

import numpy as np
import wandb
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from finetune_eval import _run_utility_evals_combined_vllm
from utils import paths as _paths

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# W&B run whose adapters / utility metrics we want to (re-)evaluate.
WANDB_ID = "RUN_022"

# Finetune step (= LoRA checkpoint index = ``utility_step`` on W&B) for the two
# checkpoints of interest.  Adapter dirs are
#   /data/new_master/lora-finetuned/<WANDB_ID>/step_<N>
step_80 = 17
step_90 = 10

# Which base model the run was finetuned on.  Pick one:
#   "llama_heretic" -> lenalibon/Meta-Llama-3-8B-Instruct-heretic-mlabonne
#                      (tokenizer overridden to the stock Llama-3-8B-Instruct
#                       one; the heretic repo ships a broken CodeLlama config)
#   "mistral"       -> mistralai/Mistral-7B-Instruct-v0.1
# RUN_022 is a Mistral run, so "mistral" is the matching choice here.
MODEL_CHOICE = "mistral"

# How many *extra* evaluation passes to run on top of the W&B-logged value.
N_EXTRA_RUNS = 4

LORA_ROOT = str(_paths.LORA_ROOT)

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = "master-thesis"

_MODEL_TABLE = {
    "llama_heretic": (
        os.environ.get("ABLITERATED_MODEL", "lenalibon/Meta-Llama-3-8B-Instruct-heretic-mlabonne"),
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "mistral": (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ),
}

CONFIG_PATH = os.path.join(_SRC_DIR, "configs", "probe_config.yaml")
FINETUNE_CONFIG_PATH = os.path.join(_SRC_DIR, "configs", "finetune_config.yaml")

UTIL_KEYS = [
    "utility/mmlu_accuracy",
    "utility/gsm8k_accuracy",
    "utility/ifeval_prompt_accuracy",
    "utility/ifeval_instruction_accuracy",
]


# ─────────────────────────── helpers ────────────────────────────────────────

def _triple_from_utility_dict(d: dict) -> dict:
    """Reduce a ``{"utility/...": float}`` dict to (mmlu, gsm8k, ifeval_avg)."""
    ifeval_prompt = d.get("utility/ifeval_prompt_accuracy")
    ifeval_instr = d.get("utility/ifeval_instruction_accuracy")
    if ifeval_prompt is None or ifeval_instr is None:
        ifeval_avg = None
    else:
        ifeval_avg = (float(ifeval_prompt) + float(ifeval_instr)) / 2.0
    mmlu = d.get("utility/mmlu_accuracy")
    gsm8k = d.get("utility/gsm8k_accuracy")
    return {
        "mmlu": float(mmlu) if mmlu is not None else None,
        "gsm8k": float(gsm8k) if gsm8k is not None else None,
        "ifeval_avg": ifeval_avg,
    }


def fetch_wandb_utility_triple(run_id: str, finetune_step: int) -> dict:
    """Pull the ``utility/*`` row logged at ``utility_step == finetune_step``."""
    if not WANDB_ENTITY:
        raise ValueError("Set WANDB_ENTITY to fetch utility metrics from W&B.")
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    rows = list(run.history(keys=UTIL_KEYS + ["utility_step"], pandas=False))
    matches = [r for r in rows if r.get("utility_step") == finetune_step]
    if not matches:
        avail = sorted(
            {r.get("utility_step") for r in rows if r.get("utility_step") is not None}
        )
        raise ValueError(
            f"No utility row for utility_step={finetune_step} in run {run_id}. "
            f"Available utility_step values: {avail}"
        )
    return _triple_from_utility_dict(matches[-1])  # last write wins if relogged


def _lora_rank(lora_path: str, fallback: int) -> int:
    """Read the LoRA rank from the adapter config so vLLM gets it right."""
    cfg_file = os.path.join(lora_path, "adapter_config.json")
    try:
        with open(cfg_file) as f:
            return int(json.load(f).get("r", fallback))
    except (OSError, ValueError, json.JSONDecodeError):
        return fallback


def run_utility_triple_once(cfg, model_name, tokenizer_name, tokenizer, lora_path) -> dict:
    """One full MMLU + GSM8K + IFEval pass via vLLM with the given LoRA adapter."""
    results = _run_utility_evals_combined_vllm(
        cfg=cfg,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        lora_path=lora_path,
        swap_model=False,
        model=None,
    )
    return _triple_from_utility_dict(results)


def _summarise(label: str, triples: list[dict]) -> None:
    print(f"\n========== {label} ==========")
    print(f"  ({len(triples)} runs: 1 from W&B + {len(triples) - 1} fresh)")
    for metric in ("mmlu", "gsm8k", "ifeval_avg"):
        vals = [t[metric] for t in triples if t.get(metric) is not None]
        if not vals:
            print(f"  {metric:11s}: no values")
            continue
        arr = np.asarray(vals, dtype=float)
        print(
            f"  {metric:11s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
            f"(values: {[round(v, 4) for v in vals]})"
        )


# ─────────────────────────────── main ───────────────────────────────────────

def main() -> None:
    if MODEL_CHOICE not in _MODEL_TABLE:
        raise ValueError(
            f"MODEL_CHOICE must be one of {list(_MODEL_TABLE)}, got {MODEL_CHOICE!r}"
        )
    model_name, tokenizer_name = _MODEL_TABLE[MODEL_CHOICE]
    print(f"Base model : {model_name}")
    print(f"Tokenizer  : {tokenizer_name}")
    print(f"W&B run    : {WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_ID}")
    print(f"Extra runs : {N_EXTRA_RUNS}")

    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_PATH),
        OmegaConf.load(FINETUNE_CONFIG_PATH),
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoints = {"step_80": step_80, "step_90": step_90}

    for label, ft_step in checkpoints.items():
        lora_path = os.path.join(LORA_ROOT, WANDB_ID, f"step_{ft_step}")
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"LoRA adapter dir not found: {lora_path}")
        cfg.lora.r = _lora_rank(lora_path, cfg.lora.r)

        print(f"\n### {label}: finetune step {ft_step}  (adapter: {lora_path})")

        triples: list[dict] = []

        wandb_triple = fetch_wandb_utility_triple(WANDB_ID, ft_step)
        print(
            f"  [wandb]  mmlu={wandb_triple['mmlu']}  gsm8k={wandb_triple['gsm8k']}  "
            f"ifeval_avg={wandb_triple['ifeval_avg']}"
        )
        triples.append(wandb_triple)

        for i in range(N_EXTRA_RUNS):
            print(f"  [run {i + 1}/{N_EXTRA_RUNS}] generating + scoring ...")
            t = run_utility_triple_once(cfg, model_name, tokenizer_name, tokenizer, lora_path)
            print(
                f"  [run {i + 1}/{N_EXTRA_RUNS}] mmlu={t['mmlu']:.4f}  "
                f"gsm8k={t['gsm8k']:.4f}  ifeval_avg={t['ifeval_avg']:.4f}"
            )
            triples.append(t)

        _summarise(f"{label} (finetune step {ft_step})", triples)


if __name__ == "__main__":
    main()
