"""Measure checkpoint monitorability with freshly fitted detectors.

Fit a linear probe and a K=16 polytope on the configured loss layer at each
selected checkpoint. Fit on BeaverTails 330k_train and evaluate on disjoint
330k_test examples, recomputing activations for each checkpoint.

Usage:
    RUN_ID=<run_id> CONFIG=<config.yaml> STEPS=1,5,10 python src/probes/refit_both_auroc.py

Writes {NM_ROOT}/live_downstream/<run_id>/refit_both/step_<N>.json with the layer
and both AUROC values. Existing step results are skipped."""
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

ROOT = os.environ.get("NM_ROOT", "/data/new_master")
RUN_ID = os.environ["RUN_ID"]
RUN_CFG = os.environ["CONFIG"]
STEPS_ENV = os.environ.get("STEPS", "all")

# Use fixed fitting hyperparameters to compare recoverability across checkpoints.
PROBE_FIT = {
    "probe": {
        "detector_type": "linear-probe",
        "normalize_input": "unit_norm",
        "train_sequence_aggregator": "mean",
        "eval_sequence_aggregator": "mean",
    },
    "probe_training": {
        "hinge_margin": 1.0, "learning_rate": 1.0e-3, "weight_decay": 1.0e-4,
        "num_epochs": int(os.environ.get("PROBE_EPOCHS", "1")), "max_steps": int(os.environ.get("PROBE_MAX_STEPS", "5000")), "batch_size": 32, "grad_accum_steps": 1,
        "clip_grad_norm": 1.0, "device": "cuda", "scheduler": "cosine",
        "warmup_steps": 50, "loss_type": "hinge",
    },
}
POLY_FIT = {
    "probe": {
        "detector_type": "polytope-probe", "normalize_input": "unit_norm",
        "num_facets": 16, "use_nonlinear": True, "polytope_max_temperature": 0.0,
        "train_sequence_aggregator": "mean", "eval_sequence_aggregator": "mean",
    },
    "probe_training": {
        "learning_rate": 1.0e-3, "weight_decay": 1.0e-4, "num_epochs": 1,
        "max_steps": 3000, "batch_size": 32, "grad_accum_steps": 1,
        "clip_grad_norm": 1.0, "device": "cuda", "scheduler": "cosine",
        "warmup_steps": 50, "loss_type": "polytope",
        "polytope": {
            "margin": 1.0, "unsafe_weight": 2.0, "f_l1_weight": 1.0e-3,
            "phi_l1_weight": 1.0e-4, "entropy_mode": "sap_reassign",
            "entropy_weight": 1.0, "entropy_temp": 1.0, "valid_edges_threshold": 0.0,
            "max_attempts": 100, "assignment_level": "token",
            "retrain_scope": "constraints_only", "freeze_input_scale": True,
        },
    },
}


def variant(base, overrides, feature_dim=None):
    """Deep copy of base cfg with probe/probe_training replaced for a fresh fit."""
    c = copy.deepcopy(base)
    ov = copy.deepcopy(overrides)
    if feature_dim is not None and "feature_dim" not in ov["probe"]:
        ov["probe"]["feature_dim"] = feature_dim
    # Replace the two sections wholesale (keys the arch config also reads, like
    # d_mlp/nhead, are inherited from probe_config.yaml under `base`).
    c.probe = OmegaConf.merge(base.probe, OmegaConf.create(ov["probe"]))
    c.probe_training = OmegaConf.merge(base.probe_training,
                                       OmegaConf.create(ov["probe_training"]))
    return c


def steps_to_do():
    stepdirs = glob.glob(os.path.join(ROOT, "lora-finetuned", RUN_ID, "step_*"))
    allsteps = sorted(int(d.split("step_")[1]) for d in stepdirs
                      if d.split("step_")[1].isdigit())
    if STEPS_ENV.strip().lower() == "all":
        return allsteps
    want = {int(x) for x in STEPS_ENV.split(",") if x.strip()}
    return [s for s in allsteps if s in want]


def main():
    wandb.init(mode="disabled")
    cfg = OmegaConf.merge(OmegaConf.load(FT.CONFIG_PATH), OmegaConf.load(RUN_CFG))
    layer = int(list(cfg.experiment.layers)[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.join(ROOT, "live_downstream", RUN_ID, "refit_both")
    os.makedirs(outdir, exist_ok=True)

    print(f"[refit-both] {RUN_ID} layer {layer}; model {cfg.model.model_name}",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    tok.pad_token = tok.eos_token
    hidden = model.config.hidden_size

    lora_layers = (list(range(model.config.num_hidden_layers))
                   if cfg.lora.layers is None else list(cfg.lora.layers))
    model = FT.initialize_lora_adapter(model, lora_layers, {
        "r": cfg.lora.r, "alpha": cfg.lora.lora_alpha, "dropout": cfg.lora.dropout,
        "target_modules": list(cfg.lora.target_modules), "task_type": "CAUSAL_LM"})
    model.eval()

    # Recompute checkpoint activations on the fixed BeaverTails 330k_test examples.
    _, _, eval_split = FT._build_probe_fitting_data(
        cfg, cfg.experiment.n_train, cfg.experiment.n_eval)

    probe_cfg = variant(cfg, PROBE_FIT)
    poly_cfg = variant(cfg, POLY_FIT, feature_dim=hidden)
    dbatch = int(cfg.probe_training.batch_size)

    for step in steps_to_do():
        out = os.path.join(outdir, f"step_{step}.json")
        if os.path.exists(out):
            print(f"[refit-both] step {step}: exists, skip", flush=True)
            continue
        adapter = os.path.join(ROOT, "lora-finetuned", RUN_ID,
                               f"step_{step}", "adapter_model.safetensors")
        if not os.path.exists(adapter):
            print(f"[refit-both] step {step}: no adapter, skip", flush=True)
            continue
        set_peft_model_state_dict(model, load_file(adapter))
        model.eval()

        res = {"step": step, "layer": layer}
        for name, fcfg in (("probe_auroc", probe_cfg), ("polytope_auroc", poly_cfg)):
            det = FT.step0_train_probes(fcfg, model, tok, [layer])[0]
            for pr in det.values():
                pr.requires_grad_(False)
                pr.eval()
            auroc = FT._refit_eval_auroc(fcfg, model, tok, det, eval_split,
                                         dbatch, device)
            res[name] = auroc
            print(f"[refit-both] step {step}: {name} = "
                  f"{auroc:.4f}" if auroc is not None else
                  f"[refit-both] step {step}: {name} = None", flush=True)

        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(res, fh)
        os.replace(tmp, out)
        print(f"[refit-both] step {step}: wrote {out}", flush=True)

    print(f"[refit-both] done {RUN_ID}", flush=True)


if __name__ == "__main__":
    main()
