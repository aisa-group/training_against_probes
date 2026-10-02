"""Measure fresh-detector AUROC across model layers and checkpoints.

Uses the fit and held-out data from refit_both_auroc. Fits both a linear probe
and a K=16 polytope at every layer. Polytope fitting defaults to 1500 steps.

Usage:
    RUN_ID=<run_id> CONFIG=<config.yaml> STEP_EVERY=5 python src/probes/refit_layer_sweep_auroc.py

Writes {NM_ROOT}/live_downstream/<run_id>/refit_layer_sweep/step_<N>.json with
per-layer AUROC for both detectors. Existing step results are skipped."""
import copy
import glob
import json
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (SRC, os.path.join(SRC, "utils"), os.path.join(SRC, "probes")):
    sys.path.insert(0, p)

import torch  # noqa: E402
import wandb  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from peft import set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import finetune as FT  # noqa: E402
from evaluate import eval_detector  # noqa: E402
from refit_both_auroc import PROBE_FIT, POLY_FIT, variant  # noqa: E402

ROOT = os.environ.get("NM_ROOT", "/data/new_master")
RUN_ID = os.environ["RUN_ID"]
RUN_CFG = os.environ["CONFIG"]
STEP_EVERY = int(os.environ.get("STEP_EVERY", "5"))
POLY_FIT = copy.deepcopy(POLY_FIT)
POLY_FIT["probe_training"]["max_steps"] = int(os.environ.get("POLY_STEPS", "1500"))


def steps_to_do():
    only = os.environ.get("STEP_ONLY", "").strip()
    if only:                                   # explicit comma-separated step list
        return [int(x) for x in only.split(",") if x.strip() != ""]
    sd = glob.glob(os.path.join(ROOT, "lora-finetuned", RUN_ID, "step_*"))
    alls = sorted(int(d.split("step_")[1]) for d in sd if d.split("step_")[1].isdigit())
    keep = [s for s in alls if s == alls[0] or s % STEP_EVERY == 0]
    # Include the base model as step 0 for each run.
    return [0] + keep


def per_layer_auroc(cfg, model, tok, detector, eval_split, dbatch, device):
    """{layer: mean held-out AUROC} for a fitted all-layer detector."""
    acc = {}
    for _pair, (ep, en) in eval_split.items():
        if not ep or not en:
            continue
        ev = eval_detector(
            detector=detector, detector_type=cfg.probe.detector_type, model=model,
            tokenizer=tok, test_dataset=(ep, en),
            obfuscate_over=cfg.experiment.obfuscate_over,
            model_batch_size=cfg.experiment.model_batch_size,
            detector_batch_size=dbatch, device=device, best_thresholds=None,
            max_sequence_length=cfg.experiment.max_sequence_length)
        for layer, m in ev["test_metrics"].items():
            a = m.get("AUC_ROC") if isinstance(m, dict) else None
            if isinstance(a, (int, float)):
                acc.setdefault(int(layer), []).append(float(a))
    return {L: sum(v) / len(v) for L, v in acc.items()}


def main():
    wandb.init(mode="disabled")
    cfg = OmegaConf.merge(OmegaConf.load(FT.CONFIG_PATH), OmegaConf.load(RUN_CFG))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.join(ROOT, "live_downstream", RUN_ID, "refit_layer_sweep")
    os.makedirs(outdir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    tok.pad_token = tok.eos_token
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    layers = list(range(n_layers))
    print(f"[layer-sweep] {RUN_ID}: {n_layers} layers, step_every={STEP_EVERY}", flush=True)

    lora_layers = (list(range(n_layers)) if cfg.lora.layers is None else list(cfg.lora.layers))
    model = FT.initialize_lora_adapter(model, lora_layers, {
        "r": cfg.lora.r, "alpha": cfg.lora.lora_alpha, "dropout": cfg.lora.dropout,
        "target_modules": list(cfg.lora.target_modules), "task_type": "CAUSAL_LM"})
    model.eval()

    # Fit/eval at ALL layers: override experiment.layers and probe_loss_layers to None.
    base = copy.deepcopy(cfg)
    base.experiment.layers = None
    base.finetune.probe_loss_layers = None
    _, _, eval_split = FT._build_probe_fitting_data(
        base, base.experiment.n_train, base.experiment.n_eval)
    probe_cfg = variant(base, PROBE_FIT)
    poly_cfg = variant(base, POLY_FIT, feature_dim=hidden)
    dbatch = int(base.probe_training.batch_size)

    _steps = steps_to_do()
    if os.environ.get("STEP_ORDER", "asc") == "desc":
        _steps = list(reversed(_steps))          # helper workers fill from the top
    for step in _steps:
        out = os.path.join(outdir, f"step_{step}.json")
        if os.path.exists(out):
            print(f"[layer-sweep] step {step}: exists, skip", flush=True)
            continue
        if step == 0:
            # Reset LoRA to its identity initialization to evaluate the base model.
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    torch.nn.init.zeros_(p)
                elif "lora_A" in n:
                    torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
            model.eval()
        else:
            adapter = os.path.join(ROOT, "lora-finetuned", RUN_ID, f"step_{step}",
                                   "adapter_model.safetensors")
            if not os.path.exists(adapter):
                continue
            set_peft_model_state_dict(model, load_file(adapter))
            model.eval()
        res = {"step": step}
        for name, fcfg in (("probe", probe_cfg), ("polytope", poly_cfg)):
            det = FT.step0_train_probes(fcfg, model, tok, layers)[0]
            for pr in det.values():
                pr.requires_grad_(False)
                pr.eval()
            res[name] = {str(L): a for L, a in
                         per_layer_auroc(fcfg, model, tok, det, eval_split, dbatch, device).items()}
            print(f"[layer-sweep] step {step}: {name} fitted at {len(res[name])} layers",
                  flush=True)
        tmp = out + f".{os.getpid()}.tmp"          # per-worker tmp: safe for parallel fill
        json.dump(res, open(tmp, "w"))
        os.replace(tmp, out)
        print(f"[layer-sweep] step {step}: wrote {out}", flush=True)
    print(f"[layer-sweep] done {RUN_ID}", flush=True)


if __name__ == "__main__":
    main()
