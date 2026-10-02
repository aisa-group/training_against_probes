"""
Probe-guided LoRA finetuning.

Steps
-----
0  Train probes on the base model (logged to W&B probe_train/ and probe_eval/).
1+ For each of n_steps model-update iterations:
     a. On-policy generation (vLLM, greedy) for harmful + benign prompts.
     b. Configured probe/polytope loss — gradients flow into LoRA weights.
     c. Token-level KL divergence loss — keeps adapted model close to base.
     d. AdamW step on LoRA parameters.
     e. Every probe_retrain_interval steps: retrain probes with frozen model.
     f. Every eval_interval steps: run finetune_eval.
"""

import gc
import json
import os
import random
import sys

# Use local HF cache — skip network freshness checks that 504 on cluster nodes.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

if os.environ.get("SOFTFILELOCK"):
    import filelock
    filelock.FileLock = filelock.SoftFileLock
    filelock.UnixFileLock = filelock.SoftFileLock

# Add probes/ and src/ to sys.path (mirrors compare_probes.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "probes"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import wandb
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import eval_detector, train_and_eval_detector
from train import DetectorArchConfig, train_detector
from utils.finetune_utils import (
    apply_chat_template,
    compute_kl_loss,
    compute_probe_loss,
    generate_on_policy_vllm,
    load_kl_dataset,
    load_training_prompts,
    sample_batch,
    tokenize_kl_batch,
    tokenize_prompt_completion_batch,
    answer_only_mask,
    thinking_length_backward,
    rollout_kl_backward,
)
from utils.jailbreak_datasets import build_paired_probe_datasets, build_probe_datasets
from utils.paths import LIVE_DOWNSTREAM, LORA_ROOT
from utils.finetune_utils import compute_polytope_lora_loss
from polytope_step_diag import dump_step_diagnostics, step_diagnostics
from utils.lora_utils import freeze_model_params, initialize_lora_adapter, save_lora_adapter, unfreeze_lora_params
from train import save_model as save_probes
from utils.wandb_utils import init_run, log_finetune_step, log_probe_eval_metrics, log_probe_train_dynamics

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "probe_config.yaml")
FINETUNE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "finetune_config.yaml")


# ──────────────────────────────── Helpers ────────────────────────────────────

def _make_train_cfg(cfg, device: str, max_steps_override: int | None = None):
    """Build a DictConfig for probe training from the merged config."""
    return OmegaConf.create(
        {
            "batch_size": cfg.probe_training.batch_size,
            "learning_rate": cfg.probe_training.learning_rate,
            "num_epochs": cfg.probe_training.num_epochs,
            "max_steps": max_steps_override if max_steps_override is not None else cfg.probe_training.max_steps,
            "weight_decay": cfg.probe_training.weight_decay,
            "grad_accum_steps": cfg.probe_training.grad_accum_steps,
            "scheduler": cfg.probe_training.scheduler,
            "warmup_steps": cfg.probe_training.warmup_steps,
            "clip_grad_norm": cfg.probe_training.clip_grad_norm,
            "device": device,
            # Probe-FITTING loss (distinct from cfg.finetune.probe_loss_type,
            # which is the LoRA-side loss). 'polytope' routes to
            # probe_archs.polytope_fit_loss with the `polytope` block below.
            "loss_type": cfg.probe_training.get("loss_type", "bce"),
            "hinge_margin": cfg.probe_training.get("hinge_margin", 1.0),
            "polytope": OmegaConf.to_container(
                cfg.probe_training.get("polytope", {}), resolve=True
            )
            or {},
        }
    )


def _make_detector_arch_config(cfg) -> DetectorArchConfig:
    return DetectorArchConfig(
        detector_type=cfg.probe.detector_type,
        normalize_input=cfg.probe.normalize_input,
        d_mlp=cfg.probe.d_mlp,
        d_proj=cfg.probe.d_proj,
        nhead=cfg.probe.nhead,
        nlayer=cfg.probe.nlayer,
        dropout=cfg.probe.dropout,
        activation=cfg.probe.activation,
        norm_first=cfg.probe.norm_first,
        train_sequence_aggregator=cfg.probe.train_sequence_aggregator,
        eval_sequence_aggregator=cfg.probe.eval_sequence_aggregator,
        num_facets=cfg.probe.get("num_facets", 16),
        feature_dim=cfg.probe.get("feature_dim", 4096),
        use_nonlinear=cfg.probe.get("use_nonlinear", True),
        polytope_max_temperature=cfg.probe.get("polytope_max_temperature", 0.0),
    )


def _build_probe_fitting_data(cfg, n_train: int, n_eval: int):
    
    """The probe/polytope fitting (off-policy, only activations are recomputed)
    
    probe_source:
      'wildguard'    (default) the configured datasets.train pairs
      'beavertails_paired'  the same, but the harmful and benign example share
                     a prompt, so topic is controlled; see the builder's docstring
      'beavertails'  PKU-Alignment/BeaverTails -- SaP's interpretability data,
                     14 harm categories
      'liars_bench'  Cadenza-Labs/liars-bench (DECEPTION, not harm): label
                     1 = deceptive, 0 = honest; config via
                     datasets.liars_bench_config (default instructed-deception)
      'harmbench'    a prepared HarmBench parquet containing the model's
                     own generations under attack, SaP's stronger setting
    """
    src = cfg.datasets.get("probe_source", "wildguard")
    if src == "beavertails":
        from utils.jailbreak_datasets import build_beavertails_probe_datasets
        return build_beavertails_probe_datasets(n_train=n_train, n_eval=n_eval)
    if src == "beavertails_paired":
        # Same corpus, but the harmful and benign example share a prompt, so the
        # detector cannot key on topic. This is also exactly the set the DPO arm
        # consumes as (prompt, chosen=safe, rejected=unsafe), which is what makes
        # the two arms comparable on data rather than only on compute.
        from utils.jailbreak_datasets import build_beavertails_paired_probe_datasets
        return build_beavertails_paired_probe_datasets(n_train=n_train, n_eval=n_eval)
    if src == "liars_bench":
        # Cadenza-Labs/liars-bench (deception). Positives = deceptive rows
        # (label 1), negatives = honest rows (label 0). prompt = folded
        # system+user turn, completion = the assistant answer. See the builder's
        # docstring for why the system instruction is folded into the prompt.
        from utils.jailbreak_datasets import build_liarsbench_probe_datasets
        return build_liarsbench_probe_datasets(
            n_train=n_train, n_eval=n_eval,
            config=cfg.datasets.get("liars_bench_config", "instructed-deception"),
        )
    if src == "harmbench":
        from utils.jailbreak_datasets import build_harmbench_probe_datasets
        return build_harmbench_probe_datasets(
            n_train=n_train, n_eval=n_eval,
            path=cfg.datasets.get("harmbench_path",
                                  "/data/new_master/datasets/harmbench_mistral.parquet"),
        )
    if src != "wildguard":
        raise ValueError(f"unknown datasets.probe_source: {src!r}")
    return build_probe_datasets(
        train_pairs=[tuple(pr) for pr in cfg.datasets.train],
        eval_pairs=[tuple(pr) for pr in cfg.datasets.eval] if n_eval else [],
        n_train=n_train, n_eval=n_eval,
    )


def _capture_diag_set(cfg, model, tokenizer, layers, device, n_per_class: int = 32):

    try:
        pos, neg, _ = build_probe_datasets(
            train_pairs=[tuple(pr) for pr in cfg.datasets.train], eval_pairs=[],
            n_train=n_per_class, n_eval=0,
        )
        n_pos, n_neg = min(n_per_class, len(pos)), min(n_per_class, len(neg))
        prompts = list(pos["prompt"])[:n_pos] + list(neg["prompt"])[:n_neg]
        comps = list(pos["completion"])[:n_pos] + list(neg["completion"])[:n_neg]
        labels = torch.tensor([1] * n_pos + [0] * n_neg)
    except Exception as e:  # noqa: BLE001
        print(f"  [diag] could not build diagnostic set ({e}); per-step diagnostics off")
        return None
    print(f"  [diag] captured {len(prompts)} sequences (text) for per-step polytope diagnostics")
    return {"prompts": prompts, "completions": comps, "labels": labels}


def _diag_activations(diag, model, tokenizer, layers, cfg, device, batch: int = 8):
    """Activations for the fixed diagnostic text, from the CURRENT model."""
    from utils.finetune_utils import tokenize_prompt_completion_batch

    acts_by_layer = {l: [] for l in layers}
    masks = []
    backbone = model.base_model.model.model if hasattr(model, "base_model") else model.model
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(diag["prompts"]), batch):
                ids, attn, comp = tokenize_prompt_completion_batch(
                    prompts=diag["prompts"][i:i + batch],
                    completions=diag["completions"][i:i + batch],
                    tokenizer=tokenizer, max_length=cfg.finetune.max_seq_len, device=device,
                )
                hs = backbone(input_ids=ids, attention_mask=attn,
                              output_hidden_states=True).hidden_states
                for l in layers:
                    acts_by_layer[l].append(hs[l + 1].to(torch.float16).cpu())
                masks.append(comp.cpu())
                del hs
    finally:
        if was_training:
            model.train()

    max_s = max(m.shape[1] for m in masks)

    def _pad(t, val=0):
        if t.shape[1] == max_s:
            return t
        pad = torch.full((t.shape[0], max_s - t.shape[1], *t.shape[2:]), val, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    mask = torch.cat([_pad(m) for m in masks], dim=0).bool()
    acts = {l: torch.cat([_pad(a) for a in acts_by_layer[l]], dim=0).float() for l in layers}
    return acts, mask


def _resolve_probe_loss_layers(cfg, detector: dict) -> list[int]:
    """Return probe loss layers from config (None → all probe layers)."""
    if cfg.finetune.probe_loss_layers is None:
        return sorted(detector.keys())
    return sorted(l for l in cfg.finetune.probe_loss_layers if l in detector)


def _resolve_polytope_layers(cfg, detector: dict) -> list[int]:
    """Layers the polytope LoRA loss is applied at.
    """
    layers = cfg.finetune.get("polytope_layers", None)
    if layers is None:
        return _resolve_probe_loss_layers(cfg, detector)
    return sorted(l for l in layers if l in detector)


# ─────────────────────────── Step 0: Probe training ──────────────────────────

def step0_train_probes(cfg, model, tokenizer, layers: list[int]) -> tuple[dict, dict | None, int, int]:
    """
    Train probes on the base model (Step 0 of finetuning).
    """
    print("\n" + "=" * 60)
    print("Step 0: Training probes on base model")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = _make_train_cfg(cfg, device)
    detector_arch_config = _make_detector_arch_config(cfg)

    train_datasets = [tuple(p) for p in cfg.datasets.train]
    eval_datasets = [tuple(p) for p in cfg.datasets.eval]

    paired_path = cfg.datasets.get("paired_path", None)
    _probe_source = cfg.datasets.get("probe_source", "wildguard")
    if _probe_source != "wildguard":
        print(f"  Probe fitting data: {_probe_source}")
        train_pos, train_neg, eval_split_datasets = _build_probe_fitting_data(
            cfg, cfg.experiment.n_train, cfg.experiment.n_eval)
    elif paired_path:
        # Paired-data ablation: fit probes on same-question harmful/benign pairs.
        train_pos, train_neg, eval_split_datasets = build_paired_probe_datasets(
            n_train=cfg.experiment.n_train,
            n_eval=cfg.experiment.n_eval,
            path=paired_path,
        )
        # Keep the configured (unpaired) eval pairs too, for comparability with
        # prior runs — one extra eval pass, an extra AUROC number for free.
        _, _, extra_eval = build_probe_datasets(
            train_pairs=[],
            eval_pairs=eval_datasets,
            n_train=cfg.experiment.n_train,
            n_eval=cfg.experiment.n_eval,
        )
        eval_split_datasets.update(extra_eval)
    else:
        train_pos, train_neg, eval_split_datasets = build_probe_datasets(
            train_pairs=train_datasets,
            eval_pairs=eval_datasets,
            n_train=cfg.experiment.n_train,
            n_eval=cfg.experiment.n_eval,
        )
    print(f"  Probe train pos: {len(train_pos)}  neg: {len(train_neg)}")

    detector: dict | None = None
    best_thresholds: dict | None = None
    all_results: dict[str, dict] = {}

    for pair_name, (eval_pos, eval_neg) in eval_split_datasets.items():
        if len(eval_pos) == 0 or len(eval_neg) == 0:
            print(f"  Skipping {pair_name}: insufficient eval data")
            continue

        print(f"\n  Evaluating pair: {pair_name}")

        if detector is None:
            # First eval pair: train probes from scratch
            _step0_cache_base = cfg.experiment.get("cache_activations_save_path", None)
            # Per-run subdir so concurrent finetune runs don't clobber each other's
            # rolling cache.
            _step0_cache = (
                os.path.join(_step0_cache_base, wandb.run.id)
                if _step0_cache_base is not None else None
            )
            if _step0_cache is not None:
                # Fresh extraction for step 0 — clear any stale cache from a prior run.
                import shutil
                shutil.rmtree(_step0_cache, ignore_errors=True)
            (
                test_metrics,
                detector,
                train_dynamics,
                _,
                best_thresholds,
            ) = train_and_eval_detector(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                train_dataset=(train_pos, train_neg),
                test_dataset=(eval_pos, eval_neg),
                train_cfg=train_cfg,
                detector_arch_config=detector_arch_config,
                obfuscate_over=cfg.experiment.obfuscate_over,
                model_batch_size=cfg.experiment.model_batch_size,
                max_sequence_length=cfg.experiment.max_sequence_length,
                cache_activations_save_path=_step0_cache,
            )
            last_dynamics_step, last_probe_step = log_probe_train_dynamics(train_dynamics)
            del train_dynamics
        else:
            # Subsequent pairs: reuse trained detector
            eval_dict = eval_detector(
                detector=detector,
                detector_type=cfg.probe.detector_type,
                model=model,
                tokenizer=tokenizer,
                test_dataset=(eval_pos, eval_neg),
                obfuscate_over=cfg.experiment.obfuscate_over,
                model_batch_size=cfg.experiment.model_batch_size,
                detector_batch_size=train_cfg.batch_size,
                device=device,
                best_thresholds=best_thresholds,
                max_sequence_length=cfg.experiment.max_sequence_length,
            )
            test_metrics = eval_dict["test_metrics"]

        all_results[pair_name] = test_metrics
        gc.collect()
        torch.cuda.empty_cache()

    eval_step = last_dynamics_step + 1
    log_probe_eval_metrics(all_results, step=eval_step)
    _assert_probe_polarity(cfg, all_results)
    return detector, best_thresholds, eval_step, last_probe_step + 1


def _assert_probe_polarity(cfg, all_results: dict) -> None:
    aucs = [
        m["AUC_ROC"]
        for layer_metrics in all_results.values()
        for m in layer_metrics.values()
        if isinstance(m, dict) and isinstance(m.get("AUC_ROC"), (int, float))
    ]
    if not aucs:
        print("  [polarity] no AUC_ROC in step-0 eval metrics; skipping polarity check")
        return
    mean_auc = sum(aucs) / len(aucs)
    print(f"  [polarity] mean step-0 probe AUC_ROC over {len(aucs)} (pair, layer) cells: {mean_auc:.4f}")
    if mean_auc < 0.5:
        raise RuntimeError(
            f"Step-0 probe AUC_ROC is {mean_auc:.4f} < 0.5, i.e. the probes are ANTI-correlated "
            f"with harmfulness. Almost certainly the loss's label branches are swapped: this repo "
            f"uses label 1 = harmful, SaP uses label 1 = safe. Check "
            f"probe_archs.polytope_fit_loss (loss_type={cfg.probe_training.get('loss_type', 'bce')!r})."
        )


# ─────────────────────────── Probe retraining ────────────────────────────────

def _refit_eval_auroc(cfg, model, tokenizer, detector, eval_split_datasets,
                      detector_batch_size, device) -> float | None:
    """Mean held-out AUC_ROC of `detector` across the eval pairs and layers.

    Evaluated on the SAME fixed, held-out eval split step 0 used (a disjoint
    dataset split from the fitting rows), so the per-step refit number is
    directly comparable to the step-0 probe AUC_ROC. AUC_ROC is threshold-free,
    so no best_thresholds are needed. Returns None if there is no usable eval
    split (e.g. logging disabled, or a run whose source ships none)."""
    if not eval_split_datasets:
        return None
    all_results: dict[str, dict] = {}
    for pair_name, (eval_pos, eval_neg) in eval_split_datasets.items():
        if len(eval_pos) == 0 or len(eval_neg) == 0:
            continue
        eval_dict = eval_detector(
            detector=detector,
            detector_type=cfg.probe.detector_type,
            model=model,
            tokenizer=tokenizer,
            test_dataset=(eval_pos, eval_neg),
            obfuscate_over=cfg.experiment.obfuscate_over,
            model_batch_size=cfg.experiment.model_batch_size,
            detector_batch_size=detector_batch_size,
            device=device,
            best_thresholds=None,
            max_sequence_length=cfg.experiment.max_sequence_length,
        )
        all_results[pair_name] = eval_dict["test_metrics"]
    aucs = [
        m["AUC_ROC"]
        for layer_metrics in all_results.values()
        for m in layer_metrics.values()
        if isinstance(m, dict) and isinstance(m.get("AUC_ROC"), (int, float))
    ]
    return sum(aucs) / len(aucs) if aucs else None


def retrain_probes(
    cfg,
    model,
    tokenizer,
    layers: list[int],
    detector: dict,
    wandb_step_offset: int,
    probe_step_offset: int,
    finetune_step: int,
) -> dict:
    """
    Retrain/reinitialise probes using the original static dataset.

    probe_retrain_mode = 'retrain': warm-start from current weights, run for probe_retrain_steps.
    probe_retrain_mode = 'reinit' : fresh probes trained from scratch using the full
                                    schedule from probe_config.yaml (probe_retrain_steps ignored).

    Returns updated detector {layer_int: Probe}.
    """
    mode = cfg.finetune.get("probe_retrain_mode", "retrain")
    if mode == "reinit":
        print(f"  [wandb_step {wandb_step_offset}] Reinitialising probes from scratch...")
    else:
        print(f"  [wandb_step {wandb_step_offset}] Retraining probes ({cfg.finetune.probe_retrain_steps} steps)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Held-out eval split for the per-step refit AUROC. Requesting n_eval>0 does
    # NOT change the fitting rows -- train and eval come from disjoint dataset
    # splits with deterministic slicing -- so refit TRAINING stays bit-identical
    # whether or not this is on. Same fixed rows step 0 used, hence comparable.
    _log_auroc = cfg.finetune.get("log_refit_auroc", True)
    _n_eval = cfg.experiment.n_eval if _log_auroc else 0

    paired_path = cfg.datasets.get("paired_path", None)
    _probe_source = cfg.datasets.get("probe_source", "wildguard")
    if _probe_source != "wildguard":
        # Same fixed text as step 0; only the activations are recomputed from the
        # current model. Off-policy by design -- see _build_probe_fitting_data.
        train_pos, train_neg, eval_split = _build_probe_fitting_data(
            cfg, cfg.experiment.n_train, _n_eval)
    elif paired_path:
        # Same paired data as step 0, deterministic slicing → identical rows.
        train_pos, train_neg, eval_split = build_paired_probe_datasets(
            n_train=cfg.experiment.n_train,
            n_eval=_n_eval,
            path=paired_path,
        )
    else:
        train_datasets = [tuple(p) for p in cfg.datasets.train]
        train_pos, train_neg, eval_split = build_probe_datasets(
            train_pairs=train_datasets,
            eval_pairs=[tuple(p) for p in cfg.datasets.eval] if _n_eval else [],
            n_train=cfg.experiment.n_train,
            n_eval=_n_eval,
        )

    if mode == "reinit":
        # Use full training schedule from probe_config.yaml; no warm-start
        train_cfg = _make_train_cfg(cfg, device, max_steps_override=None)
        extra_kwargs = {}
    else:
        train_cfg = _make_train_cfg(cfg, device, max_steps_override=cfg.finetune.probe_retrain_steps)
        pretrained = {layer: probe for layer, probe in detector.items()}
        # For a warm-started polytope, refit only phi/b by default and hold the
        # concept encoder fixed: 200 steps on n_train sequences cannot
        # meaningfully move a multi-million-parameter encoder, and a fixed
        # encoder keeps facet index k meaning the same concept across steps,
        # which is what makes per-facet drift analysis possible.
        _scope = cfg.probe_training.get("polytope", {}).get("retrain_scope", "constraints_only")
        for probe in pretrained.values():
            probe.requires_grad_(True)
            probe.train()
            if hasattr(probe, "set_retrain_scope"):
                probe.set_retrain_scope(_scope)
        extra_kwargs = {"pretrained_probes": pretrained}

    detector_arch_config = _make_detector_arch_config(cfg)

    # Rolling on-disk activation cache. Activations change every model-update
    # step (LoRA weights moved), so wipe any stale cache before extracting.
    # Per-run subdir so concurrent finetune runs don't fight over the same dir.
    _retrain_cache_base = cfg.experiment.get("cache_activations_save_path", None)
    _retrain_cache = (
        os.path.join(_retrain_cache_base, wandb.run.id)
        if _retrain_cache_base is not None else None
    )
    if _retrain_cache is not None:
        import shutil
        shutil.rmtree(_retrain_cache, ignore_errors=True)

    new_detector, _, train_dynamics = train_detector(
        model=model,
        tokenizer=tokenizer,
        train_dataset=(train_pos, train_neg),
        train_cfg=train_cfg,
        layers=layers,
        obfuscate_over=cfg.experiment.obfuscate_over,
        detector_arch_config=detector_arch_config,
        model_batch_size=cfg.experiment.model_batch_size,
        max_sequence_length=cfg.experiment.max_sequence_length,
        cache_activations_save_path=_retrain_cache,
        **extra_kwargs,
    )

    # Held-out AUROC of the freshly refit detector (same eval split as step 0),
    # logged every retrain step so the refit polytope/probe's separation is
    # visible over training, not just the step-0 value. Printed to stdout too,
    # since the Qwen runs train with W&B disabled.
    refit_auroc = None
    if _log_auroc and eval_split:
        try:
            refit_auroc = _refit_eval_auroc(
                cfg, model, tokenizer, new_detector, eval_split,
                detector_batch_size=train_cfg.batch_size, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"  [refit-auroc] eval failed at finetune step {finetune_step}: {e}")

    last_global_step, last_probe_step = log_probe_train_dynamics(
        train_dynamics,
        global_step_offset=wandb_step_offset,
        probe_step_offset=probe_step_offset,
        finetune_step=finetune_step,
    )

    if refit_auroc is not None:
        print(f"  [refit-auroc] finetune step {finetune_step}: mean held-out "
              f"refit detector AUC_ROC = {refit_auroc:.4f}")
        try:
            wandb.log({"finetune/refit_auroc": refit_auroc,
                       "finetune/refit_auroc_finetune_step": finetune_step},
                      step=last_global_step)
        except Exception:  # noqa: BLE001
            pass

    del train_dynamics
    gc.collect()
    torch.cuda.empty_cache()
    return new_detector, last_global_step, last_probe_step + 1


# ─────────────────────────── Main entry point ────────────────────────────────

def finetune():
    # ── Load config ───────────────────────────────────────────────────────────
    # FINETUNE_CONFIG env var lets a submit script point at a per-run snapshot
    # so parallel sweeps don't race on the shared YAML file.
    finetune_cfg_path = os.environ.get("FINETUNE_CONFIG", FINETUNE_CONFIG_PATH)
    print(f"Loading finetune config: {finetune_cfg_path}")
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_PATH),
        OmegaConf.load(finetune_cfg_path),
    )

    model_name = cfg.model.model_name
    tokenizer_name = cfg.model.tokenizer_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)
    torch.manual_seed(42)

    # ── Init W&B ──────────────────────────────────────────────────────────────
    run_config = OmegaConf.to_container(cfg, resolve=True)

    _mode = cfg.finetune.get("probe_retrain_mode", "retrain")
    _steps = cfg.finetune.probe_retrain_steps
    _interval = cfg.finetune.probe_retrain_interval
    _kl = cfg.finetune.kl_penalty
    _layers = cfg.finetune.probe_loss_layers
    _n_layers = len(_layers) if _layers is not None else cfg.model.get("num_hidden_layers", "all")

    _kl_str = str(int(_kl)) if _kl == int(_kl) else str(_kl)
    if _mode == "reinit":
        _retrain_str = f"reinit.{_interval}"
    elif _steps == 0:
        _retrain_str = "noretrain"
    else:
        _retrain_str = f"retrain{_steps}.{_interval}"

    _ifeval_loss_cfg = cfg.finetune.get("ifeval_loss", None)
    _ifeval_loss_enabled = _ifeval_loss_cfg is not None and _ifeval_loss_cfg.get("enabled", False)

    _model_short = cfg.model.model_name.split("/")[-1]
    _probe_loss_type = cfg.finetune.get("probe_loss_type", "bce")
    _hinge_margin = cfg.finetune.get("probe_hinge_margin", 1.0)
    _run_name = f"{_model_short}_{_retrain_str}_kl{_kl_str}_{_n_layers}layers"
    if _ifeval_loss_enabled:
        _run_name += "_ifeval"
    if _probe_loss_type == "polytope":
        _K = cfg.probe.get("num_facets", 16)
        _F = cfg.probe.get("feature_dim", 4096)
        _run_name += f"_poly{_K}f{_F}"
        # The LoRA-side loss must appear in the name: without it the hinge and
        # BCE arms of the grid produce identical run names and collide in W&B.
        _run_name += f"_{cfg.finetune.get('polytope_lora_loss', 'sum_relu')}"
        _run_name += f"_{cfg.datasets.get('probe_source', 'wildguard')}"
        if not cfg.probe.get("use_nonlinear", True):
            _run_name += "_linenc"
    elif _probe_loss_type != "bce":
        _run_name += f"_{_probe_loss_type}{_hinge_margin}"
    _tags = [f"model_{_model_short}", f"mode_{_mode}", f"kl_{_kl_str}", f"layers_{_n_layers}", f"interval_{_interval}", f"loss_{_probe_loss_type}"]
    if _probe_loss_type == "polytope":
        _tags += [
            f"K_{cfg.probe.get('num_facets', 16)}",
            f"F_{cfg.probe.get('feature_dim', 4096)}",
            f"polyloss_{cfg.finetune.get('polytope_lora_loss', 'sum_relu')}",
            "nonlinear" if cfg.probe.get("use_nonlinear", True) else "linear_encoder",
        ]
    if _ifeval_loss_enabled:
        _tags.append("ifeval_loss")

    # Allow an external launcher to pre-generate the W&B run id so sibling
    # cluster jobs (e.g. downstream_eval.py) can resume into the same
    # run with `resume="allow"` and write per-step downstream metrics in
    # parallel.
    _wandb_run_id = os.environ.get("WANDB_RUN_ID") or None
    init_run(config=run_config, name=_run_name, tags=_tags, id=_wandb_run_id)
    # Mark step 0 probe training clearly in W&B
    wandb.log({"phase": "step0_probe_train"})

    # ── Per-run directories (keyed by W&B run ID) ────────────────────────────
    save_dir = os.path.join(cfg.finetune.save_dir, wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)
    lora_all_ckpts_dir = os.path.join(str(LORA_ROOT), wandb.run.id)
    os.makedirs(lora_all_ckpts_dir, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    # Pick up from the newest complete checkpoint of THIS run id. A 30-step run
    # is ~14 h, and the cluster removed a whole batch mid-flight once already;
    # without this every such event costs the entire run rather than the tail.
    #
    # What is restored: the LoRA weights, the detector (refit every step, so its
    # state at step N is not reproducible from step 0), and the step-0 baseline
    # utility that the STOP gate compares against. What is NOT restored is the
    # AdamW moments, which are not written to disk -- 130M trainable parameters
    # would be ~1 GB per step. A resumed run therefore restarts momentum from
    # zero, which is a real but bounded discontinuity: the LoRA learning rate is
    # constant, so nothing else falls out of step.
    _resume_step = 0
    if cfg.finetune.get("resume", True):
        _cands = []
        for d in os.listdir(lora_all_ckpts_dir):
            if not d.startswith("step_") or not d.split("_")[1].isdigit():
                continue
            sd = os.path.join(lora_all_ckpts_dir, d)
            if (os.path.exists(os.path.join(sd, "adapter_config.json"))
                    and os.path.isdir(os.path.join(sd, "probes"))):
                _cands.append(int(d.split("_")[1]))
        if _cands:
            _resume_step = max(_cands)
    _resume_dir = (os.path.join(lora_all_ckpts_dir, f"step_{_resume_step}")
                   if _resume_step else None)
    if _resume_dir:
        print(f"\n[resume] found step_{_resume_step} for run {wandb.run.id}; "
              f"continuing from there (steps 1-{_resume_step} are kept)")

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    _n_layers = model.config.num_hidden_layers
    _probe_layers = cfg.finetune.probe_loss_layers
    if _probe_layers is not None:
        _bad = [l for l in _probe_layers if l < 0 or l >= _n_layers]
        if _bad:
            raise ValueError(
                f"probe_loss_layers contains layer indices {_bad} that are out "
                f"of range for {model_name} ({_n_layers} hidden layers). Valid "
                f"range is [0, {_n_layers - 1}]. Update finetune_config.yaml."
            )

    if cfg.experiment.layers is not None:
        layers = list(cfg.experiment.layers)
    else:
        layers = list(range(model.config.num_hidden_layers))

    # ── Step 0: Train probes on base model ────────────────────────────────────
    if _resume_dir:
        # The detector is refit every step under the continuously-updated regime,
        # so its state at step N is a function of the whole trajectory and cannot
        # be recovered by refitting on the base model. Load the saved one.
        from train import load_model as load_probes
        detector = load_probes(os.path.join(_resume_dir, "probes"))
        best_thresholds = None       # only ever used inside step0_train_probes
        wb_step = probe_step = 0
        print(f"[resume] loaded detector from {_resume_dir}/probes "
              f"({len(detector)} layer(s))")
    else:
        detector, best_thresholds, wb_step, probe_step = step0_train_probes(
            cfg, model, tokenizer, layers)

    # Freeze probe parameters for the finetuning phase
    for probe in detector.values():
        probe.requires_grad_(False)
        probe.eval()

    # ── Apply LoRA adapters ───────────────────────────────────────────────────
    print("\nApplying LoRA adapters...")
    lora_layers = (
        list(range(model.config.num_hidden_layers))
        if cfg.lora.layers is None
        else list(cfg.lora.layers)
    )
    lora_params = {
        "r": cfg.lora.r,
        "alpha": cfg.lora.lora_alpha,
        "dropout": cfg.lora.dropout,
        "target_modules": list(cfg.lora.target_modules),
        "task_type": "CAUSAL_LM",
    }
    model = initialize_lora_adapter(model, lora_layers, lora_params)
    if _resume_dir:
        # Weights are loaded into the freshly-built adapter rather than via
        # PeftModel.from_pretrained, so the rest of the setup (layer selection,
        # requires_grad, gradient checkpointing) stays on the one code path.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        _sd = load_file(os.path.join(_resume_dir, "adapter_model.safetensors"))
        _res = set_peft_model_state_dict(model, _sd)
        _missing = getattr(_res, "unexpected_keys", None)
        print(f"[resume] loaded LoRA weights from {_resume_dir}"
              + (f" (unexpected keys: {len(_missing)})" if _missing else ""))

    # Gradient checkpointing: recompute activations during backward instead of
    # storing them. Without this, a batch-16 × seq-1024 forward through 32 layers
    # of LLaMA-3-8B exceeds 79 GB just in intermediate activations.
    # enable_input_require_grads() is required so LoRA grads flow through
    # checkpointed segments (LoRA inputs don't require_grad by default).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("  Gradient checkpointing enabled.")

    # ── Optimizer (LoRA weights only) ─────────────────────────────────────────
    lora_trainable = [p for name, p in model.named_parameters() if p.requires_grad]
    print(f"  Trainable LoRA parameters: {sum(p.numel() for p in lora_trainable):,}")
    # Qwen3 reasons by default and every other model here ignores the switch.
    # Set once, process-wide, so the on-policy generation, the KL anchor and
    # every downstream eval agree; see finetune_utils.apply_chat_template for
    # why this used to be hardcoded off.
    from utils.finetune_utils import set_enable_thinking
    _pool = cfg.finetune.get("probe_loss_pooling", "mean")
    print(f"[config] probe_loss_pooling={_pool} "
          f"(token = per-token hinge then mean; mean = pool then hinge)")
    _think = bool(cfg.finetune.get("enable_thinking", False))
    set_enable_thinking(_think)
    print(f"[config] enable_thinking={_think}")

    optimizer = torch.optim.AdamW(
        lora_trainable,
        lr=cfg.finetune.lr,
        weight_decay=cfg.finetune.weight_decay,
    )

    # ── Load prompt pools ─────────────────────────────────────────────────────
    print("\nLoading training prompt pools...")
    harmful_pool = load_training_prompts(cfg.finetune.harmful_train_split)
    benign_pool = load_training_prompts(cfg.finetune.benign_train_split)
    print(f"  Harmful pool: {len(harmful_pool)}  Benign pool: {len(benign_pool)}")

    # Two mutually exclusive maths anchors. Prompt-only (kl_dataset_math) scores
    # the model while it READS a problem; rollout (kl_rollouts_math) scores it
    # over a worked solution the un-finetuned model produced, which is the only
    # one of the two that ever observes a reasoning chain. A run uses at most one,
    # so the arms differ in one thing.
    _math_kl_rollouts = []
    if cfg.finetune.get("kl_rollouts_math", None):
        if cfg.finetune.get("kl_dataset_math", None):
            raise ValueError(
                "set kl_rollouts_math OR kl_dataset_math, not both: they are two "
                "versions of the same maths anchor and running both would make the "
                "arm uninterpretable")
        from utils.finetune_utils import load_kl_rollouts
        _math_kl_rollouts = load_kl_rollouts(
            cfg.finetune.kl_rollouts_math,
            max_n=cfg.finetune.get("kl_max_pool_math", 1000),
        )
        _lens = [len(c.split()) for _, c in _math_kl_rollouts]
        print(f"  Maths KL rollouts: {len(_math_kl_rollouts)} (prompt, solution) "
              f"pairs ({cfg.finetune.kl_rollouts_math}); "
              f"solution length mean {sum(_lens) / len(_lens):.0f} words")

    _math_kl_pool = []
    if cfg.finetune.get("kl_dataset_math", None):
        from utils.finetune_utils import load_math_kl_dataset
        _math_kl_pool = load_math_kl_dataset(
            cfg.finetune.kl_dataset_math,
            max_n=cfg.finetune.get("kl_max_pool_math", 1000),
        )
        print(f"  Maths KL pool: {len(_math_kl_pool)} prompts "
              f"({cfg.finetune.kl_dataset_math})")
        if not _math_kl_pool:
            raise ValueError(
                f"kl_dataset_math={cfg.finetune.kl_dataset_math} is configured but "
                "loaded 0 prompts; refusing to train with the maths anchor silently "
                "disabled")

    print("Loading KL divergence dataset...")
    # The main anchor takes the same prompt-or-rollout choice as the maths one.
    _kl_rollouts = []
    kl_pool = []
    if cfg.finetune.get("kl_rollouts", None):
        if cfg.finetune.get("kl_dataset", None):
            raise ValueError(
                "set kl_rollouts OR kl_dataset, not both: they are two versions "
                "of the same anchor")
        from utils.finetune_utils import load_kl_rollouts
        _kl_rollouts = load_kl_rollouts(cfg.finetune.kl_rollouts,
                                      max_n=cfg.finetune.kl_max_pool)
        _l = [len(c.split()) for _, c in _kl_rollouts]
        print(f"  KL rollouts: {len(_kl_rollouts)} (prompt, completion) pairs "
              f"({cfg.finetune.kl_rollouts}); completion mean {sum(_l)/len(_l):.0f} words")
    else:
        kl_pool = load_kl_dataset(cfg.finetune.kl_dataset,
                                  max_n=cfg.finetune.kl_max_pool)
        print(f"  KL pool: {len(kl_pool)}")

    probe_loss_layers = _resolve_probe_loss_layers(cfg, detector)
    print(f"  Probe loss layers ({len(probe_loss_layers)}): {probe_loss_layers}")

    polytope_layers = _resolve_polytope_layers(cfg, detector)

    # Fixed activation set for per-step polytope diagnostics. Captured ONCE from
    # the base model so every step's numbers are comparable to each other and
    # across runs -- if it were recaptured each step, a change in the metric
    # could be the model moving or the eval set moving, and we could not tell.
    _diag = None
    if cfg.finetune.get("probe_loss_type", "bce") == "polytope" and cfg.finetune.get(
            "polytope_step_diagnostics", True):
        _diag = _capture_diag_set(cfg, model, tokenizer, polytope_layers, device)

    if cfg.finetune.get("probe_loss_type", "bce") == "polytope":
        print(f"  Polytope LoRA-loss layers ({len(polytope_layers)}): {polytope_layers}")
        print(f"  Polytope LoRA-loss mode: {cfg.finetune.get('polytope_lora_loss', 'sum_relu')}, "
              f"margin={cfg.finetune.get('polytope_lora_margin', 1.0)}")

    ifeval_train_pool: list = []
    if _ifeval_loss_enabled:
        from finetune_eval import load_ifeval_train_pool
        n_eval_skip = cfg.utility_eval.ifeval_n_samples
        ifeval_train_pool = load_ifeval_train_pool(n_eval_skip)
        print(f"  IFEval loss train pool: {len(ifeval_train_pool)} samples (skipped first {n_eval_skip} for utility eval)")

    # Persistent temp dir for LoRA adapter checkpoints
    lora_ckpt_path = os.path.join(save_dir, "lora_adapter_current")
    swap_for_vllm: bool = cfg.finetune.get("swap_model_for_vllm", True)
    inmemory_onpolicy: bool = cfg.finetune.get("use_inmemory_for_onpolicy", False)

    # ── Finetuning loop ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Finetuning: {cfg.finetune.n_steps} steps")
    print(f"{'=' * 60}")

    wb_step += 1
    wandb.log({"phase": "finetune"})
    global_step = 0

    # Baseline: eval the abliterated model before any finetuning
    from finetune_eval import run_eval  # noqa: E402
    _baseline_json = LIVE_DOWNSTREAM / wandb.run.id / "baseline_utility.json"
    if _resume_dir and _baseline_json.exists():
        # The baseline is the UN-finetuned model, a fixed quantity for this run.
        # Re-running it here would score the RESUMED weights instead and move the
        # STOP threshold, so a resumed run would be gated against itself.
        _b = json.loads(_baseline_json.read_text())
        _baseline_utility_results = {
            "utility/mmlu_accuracy": _b["mmlu"],
            "utility/gsm8k_accuracy": _b["gsm8k"],
            "utility/ifeval_prompt_accuracy": _b["ifeval_prompt"],
            "utility/ifeval_instruction_accuracy": _b["ifeval_instr"],
        }
        print(f"[resume] reusing the step-0 baseline from {_baseline_json.name} "
              f"(aggregate {_b['aggregate']:.4f})")
    else:
        print("\nRunning baseline evaluation on abliterated model...")
        save_lora_adapter(model, lora_ckpt_path)
        model.eval()
        wb_step += 1
        _baseline_utility_results = run_eval(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            detector=detector,
            layers=layers,
            lora_path=lora_ckpt_path,
            global_step=wb_step,
            finetune_step=0,
        )
        model.train()

    # Held-out downstream eval (StrongREJECT) runs in-process.
    # It used to be a sibling condor job polling for checkpoints, which meant two
    # GPUs per run -- ~144 for the 72-run grid, enough that the scheduler packed
    # six of our jobs onto one node.
    #
    # Utility is NOT part of it: run_eval above already measures MMLU/GSM8K/IFEval
    # every step from the live in-memory model, under the same utility/* keys. The
    # watcher's utility pass was a second, slower measurement of the same thing.
    # The baseline for STOP-on-collapse is run_eval's own step-0 result, which is
    # the un-finetuned model (LoRA is identity at init, B=0).
    from asr.downstream_eval import aggregate_utility  # noqa: E402
    _baseline_utility = aggregate_utility(_baseline_utility_results or {})
    _utility_threshold = (
        cfg.finetune.get("utility_threshold_pct", 0.8) * _baseline_utility["aggregate"]
        if _baseline_utility_results else None
    )
    if _utility_threshold is not None:
        print(f"  [utility] baseline aggregate={_baseline_utility['aggregate']:.4f}  "
              f"STOP threshold={_utility_threshold:.4f}", flush=True)
        wandb.log({
            "utility/baseline_mmlu": _baseline_utility["mmlu"],
            "utility/baseline_gsm8k": _baseline_utility["gsm8k"],
            "utility/baseline_ifeval_avg": _baseline_utility["ifeval_avg"],
            "utility/baseline_aggregate": _baseline_utility["aggregate"],
            "utility/threshold": _utility_threshold,
        })
        # Persist to /fast as well as W&B -- the watcher used to do this, and the
        # per-step breakdown is what the offline utility-threshold analysis reads.
        _util_dir = LIVE_DOWNSTREAM / wandb.run.id
        _util_dir.mkdir(parents=True, exist_ok=True)
        (_util_dir / "baseline_utility.json").write_text(
            json.dumps(_baseline_utility, indent=2))

    # Recorded into every utility JSON: "lm_eval" only when the run actually
    # routes there, which needs BOTH backend lm_eval and use_inmemory_model false
    # -- the in-memory branch never reaches the backend switch at all.
    _utility_scorer = (
        str(getattr(cfg.utility_eval, "backend", None) or "lm_eval").lower()
        if not getattr(cfg.utility_eval, "use_inmemory_model", False)
        else "builtin_inmemory")

    _downstream = None
    if cfg.finetune.get("inline_downstream", True):
        from asr.downstream_inline import InlineDownstream  # noqa: E402
        _downstream = InlineDownstream(
            tokenizer=tokenizer, run_id=wandb.run.id,
            base_model=model_name, base_tokenizer=tokenizer_name,
            max_new_tokens=cfg.finetune.max_new_tokens,
        )
    _stop_utility_collapsed = False

    for step in range(_resume_step, cfg.finetune.n_steps):
        if _stop_utility_collapsed:
            print(f"\n[STOP] Aggregate utility fell below threshold; stopping at "
                  f"step {step}.")
            break
        model.train()
        optimizer.zero_grad()

        # ── 1. Sample harmful + benign prompts ────────────────────────────────
        harmful_prompts = sample_batch(harmful_pool, cfg.finetune.batch_size)
        benign_prompts = sample_batch(benign_pool, cfg.finetune.batch_size)
        main_prompts = harmful_prompts + benign_prompts

        # Sample IFEval prompts for the loss term (generated in the same vLLM call)
        if _ifeval_loss_enabled:
            _n_ifeval = _ifeval_loss_cfg.n_samples
            _ifeval_batch = random.sample(ifeval_train_pool, min(_n_ifeval, len(ifeval_train_pool)))
            _ifeval_prompts = [item[0] for item in _ifeval_batch]
            _ifeval_instr_ids = [item[1] for item in _ifeval_batch]
            _ifeval_kwargs = [item[2] for item in _ifeval_batch]
            gen_prompts = main_prompts + _ifeval_prompts
        else:
            gen_prompts = main_prompts

        # ── 2. On-policy generation ───────────────────────────────────────────
        # ── 2. On-policy generation ───────────────────────────────────────────
        formatted = apply_chat_template(gen_prompts, tokenizer)
        if inmemory_onpolicy:
            # Use the live PEFT model directly — no subprocess, no disk reload.
            # For Qwen3 the vLLM path falls back to HF anyway and pays for two
            # model loads per step; in-memory skips both.
            _think_budget = cfg.finetune.get("thinking_budget", None)
            if _think_budget:
                # Budget forcing: cap the reasoning at thinking_budget tokens and
                # inject </think> so every completion terminates and has an answer
                # (probe/StrongREJECT stay answer-only).
                from finetune_eval import _generate_inmemory_budget
                all_completions = _generate_inmemory_budget(
                    model, tokenizer, formatted,
                    think_budget=int(_think_budget),
                    answer_budget=int(cfg.finetune.get("answer_budget", 512)),
                )
            else:
                from finetune_eval import _generate_inmemory
                all_completions = _generate_inmemory(
                    model, tokenizer, formatted,
                    max_new_tokens=cfg.finetune.max_new_tokens,
                )
        else:
            # Save adapter, optionally move model to CPU, generate via vLLM, restore.
            save_lora_adapter(model, lora_ckpt_path)
            if swap_for_vllm:
                model.to("cpu")
                torch.cuda.empty_cache()

            all_completions = generate_on_policy_vllm(
                formatted_prompts=formatted,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                lora_path=lora_ckpt_path,
                max_new_tokens=cfg.finetune.max_new_tokens,
                gpu_memory_utilization=cfg.finetune.vllm_gpu_memory_utilization,
                lora_rank=cfg.lora.r,
            )

            if swap_for_vllm:
                model.to(device)
        model.train()

        n_main = len(main_prompts)
        completions = all_completions[:n_main]
        if _ifeval_loss_enabled:
            _ifeval_completions = all_completions[n_main:]

        # ── 3. Is the model's OWN output actually getting safer? ──────────────
        # The polytope diagnostics measure representation drift on fixed text --
        # deliberately, because that isolates obfuscation. They say nothing about
        # what the model actually produces. This scores the on-policy harmful
        # completions the LoRA is being trained on, so the two can be read
        # together: if the polytope score falls while this does not, the model
        # learned to hide rather than to refuse.
        #
        _onpol_sr = None
        _onpol_rows = None
        if cfg.finetune.get("score_onpolicy_strongreject", True):
            try:
                import numpy as _np2  # noqa: E402

                from asr.judges import score_strongreject_batch as _sr_now  # noqa: E402
                _n_harm = len(harmful_prompts)
                # Score the ANSWER only on a reasoning run, never the trace --
                # same rule the detector loss follows (answer_only_mask). The
                # completion is <thinking></think><answer>, so the answer is the
                # text after the last </think>. Non-reasoning runs keep the full
                # completion (they have no </think> to split on).
                if _think:
                    from utils.finetune_utils import answer_only_text
                    _sr_answers = [answer_only_text(c) for c in completions[:_n_harm]]
                else:
                    _sr_answers = list(completions[:_n_harm])
                _prev = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
                os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["HF_DATASETS_OFFLINE"] = "0"
                try:
                    _scores = _sr_now(harmful_prompts, _sr_answers)
                finally:
                    for _k, _v in _prev.items():
                        if _v is None:
                            os.environ.pop(_k, None)
                        else:
                            os.environ[_k] = _v
                _onpol_sr = float(_np2.mean(_scores))
                wandb.log({
                    "finetune/onpolicy_harmful_strongreject": _onpol_sr,
                    "finetune/onpolicy_harmful_asr_ge_0_5": float(_np2.mean([x >= 0.5 for x in _scores])),
                })
                print(f"    [safety] on-policy harmful StrongREJECT = {_onpol_sr:.4f}")
                # Keep the generations themselves, not just the score. A number
                # cannot tell you WHY it moved: a model that starts refusing, one
                # that answers evasively, and one that degenerates into gibberish
                # all drive StrongREJECT down, and only the text distinguishes
                # them. Saved next to the step's adapter so it can be read later
                # without re-running anything.
                # Keep the FULL completion (trace included) for WHY-analysis,
                # but record the answer-only text StrongREJECT actually scored so
                # the number and the text it grades are never out of step.
                _onpol_rows = [
                    {"step": step + 1, "prompt": _pr, "completion": _co,
                     "answer": _an, "strongreject": float(_sc), "label": "harmful"}
                    for _pr, _co, _an, _sc in zip(
                        harmful_prompts, completions[:_n_harm], _sr_answers, _scores)
                ]
                _benign_pr = main_prompts[_n_harm:]
                _onpol_rows += [
                    {"step": step + 1, "prompt": _pr, "completion": _co,
                     "strongreject": None, "label": "benign"}
                    for _pr, _co in zip(_benign_pr, completions[_n_harm:])
                ]
            except Exception as e:  # noqa: BLE001
                print(f"    [safety] on-policy StrongREJECT failed: {e}")

        # ── 3. Probe loss ─────────────────────────────────────────────────────
        # Gradient: loss → probe (frozen) → activations → LoRA
        input_ids, attention_mask, completion_mask = tokenize_prompt_completion_batch(
            prompts=main_prompts,
            completions=completions,
            tokenizer=tokenizer,
            max_length=cfg.finetune.max_seq_len,
            device=device,
        )
        # On a reasoning model the detector scores the ANSWER only, never the
        # thinking trace. Scoring the trace would train the model to keep its
        # answer while rewriting the reasoning to look benign -- obfuscated
        # chain of thought, manufactured by our own objective -- and would
        # destroy the monitorability the trace exists for. The KL anchor keeps
        # the FULL completion, trace included, because that is the capability
        # being preserved. Both still attend to everything; only the scored
        # positions differ.
        # By default on a reasoning model the detector scores the ANSWER only,
        # never the thinking trace (answer_only_mask) -- scoring the trace risks
        # training the model to rewrite its reasoning to look benign (obfuscated
        # CoT). detector_include_thinking flips that on purpose: the harmful
        # decision is made IN the trace, so scoring the full completion (trace +
        # answer) puts the detector loss directly on where harm is decided, at
        # the cost of that CoT-obfuscation risk. Both still attend to everything.
        detector_mask = completion_mask
        if _think and not cfg.finetune.get("detector_include_thinking", False):
            detector_mask, _n_noans = answer_only_mask(
                input_ids, completion_mask, tokenizer)
            _kept = int(detector_mask.sum())
            _all = int(completion_mask.sum())
            print(f"    [detector] scoring answer only: {_kept}/{_all} completion "
                  f"tokens ({_kept / max(_all, 1):.0%}); {_n_noans}/"
                  f"{completion_mask.shape[0]} generations had no closed "
                  f"</think> and contribute nothing")
            if _kept == 0:
                print("    [detector] WARNING: no answer tokens in this batch; "
                      "the detector term is exactly zero this step")
        elif _think:
            print(f"    [detector] scoring FULL completion (thinking + answer): "
                  f"{int(completion_mask.sum())} tokens "
                  f"(detector_include_thinking=true)")

        if cfg.finetune.get("probe_loss_type", "bce") == "polytope":
            probe_loss = compute_polytope_lora_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                completion_mask=detector_mask,
                probes=detector,
                polytope_layers=polytope_layers,
                margin=cfg.finetune.get("polytope_lora_margin", 1.0),
                mode=cfg.finetune.get("polytope_lora_loss", "sum_relu"),
                lse_temp=cfg.finetune.get("polytope_lse_temp", 1.0),
            )
        else:
            probe_loss = compute_probe_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                completion_mask=detector_mask,
                probes=detector,
                probe_loss_layers=probe_loss_layers,
                loss_type=cfg.finetune.get("probe_loss_type", "bce"),
                hinge_margin=cfg.finetune.get("probe_hinge_margin", 1.0),
                pooling=cfg.finetune.get("probe_loss_pooling", "mean"),
            )
        p_loss_val = probe_loss.item()
        # Weight on the detector (probe/polytope) term. Default 1.0 leaves every
        # existing run unchanged; a value > 1 pushes the LoRA harder to satisfy
        # the detector relative to the KL anchor. p_loss_val stays the RAW loss so
        # the printed "probe=" is comparable across runs; only the gradient scales.
        _det_w = float(cfg.finetune.get("probe_loss_weight", 1.0))
        (_det_w * probe_loss if _det_w != 1.0 else probe_loss).backward()
        del probe_loss
        gc.collect()
        torch.cuda.empty_cache()

        # ── 3b. Thinking-length penalty (reasoning models) ────────────────────
        # A small, SEPARATE term that only shortens the reasoning trace: it
        # encourages </think> throughout the thinking span (raises p(</think>)),
        # so the model terminates its trace sooner. The detector is untouched --
        # it still scores the ANSWER only (detector_mask), never the reasoning.
        # This exists because the answer-only objective + KL was found to lower
        # answer-harmfulness partly by looping in the trace until the token limit
        # (never emitting </think>), which is degenerate, not genuine refusal.
        len_pen_val = None
        _len_pen = float(cfg.finetune.get("thinking_length_penalty", 0.0))
        if _think and _len_pen > 0:
            think_mask = completion_mask.bool() & ~detector_mask.bool()
            _end_id = tokenizer.convert_tokens_to_ids("</think>")
            len_pen_val = thinking_length_backward(
                model, input_ids, attention_mask, think_mask, _end_id,
                penalty=_len_pen,
                micro_batch=cfg.finetune.get("kl_micro_batch", 8))
            print(f"    [len] thinking-length penalty lambda={_len_pen}: "
                  f"mean -log p(</think>) = {len_pen_val:.4f} on "
                  f"{int(think_mask.sum())} thinking tokens", flush=True)
            gc.collect(); torch.cuda.empty_cache()

        # ── 4. KL divergence loss ─────────────────────────────────────────────
        if _kl_rollouts:
            idx = random.sample(range(len(_kl_rollouts)),
                                min(cfg.finetune.batch_size_kl, len(_kl_rollouts)))
            kl_val = rollout_kl_backward(
                model, [_kl_rollouts[i] for i in idx], tokenizer,
                penalty=cfg.finetune.kl_penalty,
                max_length=cfg.finetune.get("kl_max_seq_len_roll", 512),
                micro_batch=cfg.finetune.get("kl_micro_batch", 8),
                chunk_size=cfg.finetune.kl_chunk_size, device=device,
            )
        else:
            kl_prompts = sample_batch(kl_pool, cfg.finetune.batch_size_kl)
            kl_ids, kl_mask = tokenize_kl_batch(
                prompts=kl_prompts,
                tokenizer=tokenizer,
                max_length=cfg.finetune.kl_max_seq_len,
                device=device,
            )
            kl_loss = compute_kl_loss(
                model=model,
                input_ids=kl_ids,
                attention_mask=kl_mask,
                chunk_size=cfg.finetune.kl_chunk_size,
            )
            kl_val = kl_loss.item()
            (cfg.finetune.kl_penalty * kl_loss).backward()

        # ── 4a. Maths KL anchor (optional) ───────────────────────────────────
        # A second KL term on held-out maths prompts. The UltraChat anchor is
        # conversational, so nothing in the objective pushes back when the probe
        # loss erodes multi-step arithmetic; this term gives that its own anchor.
        math_kl_val = None
        if _math_kl_pool:
            math_prompts = sample_batch(_math_kl_pool, cfg.finetune.batch_size_kl_math)
            m_ids, m_mask = tokenize_kl_batch(
                prompts=math_prompts,
                tokenizer=tokenizer,
                max_length=cfg.finetune.kl_max_seq_len,
                device=device,
            )
            math_kl = compute_kl_loss(
                model=model, input_ids=m_ids, attention_mask=m_mask,
                chunk_size=cfg.finetune.kl_chunk_size,
            )
            math_kl_val = float(math_kl)
            (cfg.finetune.kl_penalty_math * math_kl).backward()
            del math_kl

        # ── 4a-rollout. Maths KL over the base model's own solutions ──────────
        # KL( p_base( . | x, s_<t) || p_current( . | x, s_<t) ) averaged over the
        # completion tokens of s, where s was generated once by the un-finetuned
        # model and is now frozen. The prompt-only anchor above only ever sees the
        # model read a problem; this one watches it solve one, which is where the
        # multi-step arithmetic that GSM8K measures actually lives.
        #
        # Micro-batched: prompt+solution is ~2x the length of a prompt, so a full
        # batch would double peak memory against two full-vocab forwards. Splitting
        # keeps the same number of prompts per step at the peak memory of the
        # smaller batch. Scaling each part by its share of the scored tokens and
        # accumulating is identical to one large batch, because the terms share the
        # grad buffers until the single optimizer.step().
        elif _math_kl_rollouts:
            idx = random.sample(range(len(_math_kl_rollouts)),
                                min(cfg.finetune.batch_size_kl_math,
                                    len(_math_kl_rollouts)))
            math_kl_val = rollout_kl_backward(
                model, [_math_kl_rollouts[i] for i in idx], tokenizer,
                penalty=cfg.finetune.kl_penalty_math,
                max_length=cfg.finetune.get("kl_max_seq_len_math", 512),
                micro_batch=cfg.finetune.get("kl_micro_batch_math", 8),
                chunk_size=cfg.finetune.kl_chunk_size, device=device,
            )

        # ── 4b. IFEval instruction-following loss (optional) ──────────────────
        if_loss_val = if_prompt_acc = if_instr_acc = None
        if _ifeval_loss_enabled:
            from finetune_eval import compute_ifeval_loss
            ifeval_loss, if_prompt_acc, if_instr_acc = compute_ifeval_loss(
                model=model,
                prompts=_ifeval_prompts,
                completions=_ifeval_completions,
                instr_ids_list=_ifeval_instr_ids,
                kwargs_list=_ifeval_kwargs,
                tokenizer=tokenizer,
                max_seq_len=_ifeval_loss_cfg.max_seq_len,
                device=device,
            )
            if_loss_val = ifeval_loss.item()
            (_ifeval_loss_cfg.penalty * ifeval_loss).backward()
            del ifeval_loss
            gc.collect()
            torch.cuda.empty_cache()

        # ── 5. Gradient step ──────────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(lora_trainable, 1.0)
        optimizer.step()
        global_step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        p_loss = p_loss_val
        k_loss = kl_val
        # t_loss reflects what is actually optimised: the detector term enters the
        # gradient scaled by probe_loss_weight (default 1.0), so scale it here too.
        t_loss = _det_w * p_loss + cfg.finetune.kl_penalty * k_loss
        # The maths anchor belongs in the total like every other term. It was
        # computed and dropped on the floor before: neither logged nor summed, so
        # a run with a maths anchor reported the same total as one without.
        if math_kl_val is not None:
            t_loss += cfg.finetune.kl_penalty_math * math_kl_val
        if if_loss_val is not None:
            t_loss += _ifeval_loss_cfg.penalty * if_loss_val
        if len_pen_val is not None:
            t_loss += _len_pen * len_pen_val
        wb_step += 1
        log_finetune_step(
            wb_step, p_loss, k_loss, t_loss,
            ifeval_loss=if_loss_val,
            ifeval_prompt_accuracy=if_prompt_acc,
            ifeval_instruction_accuracy=if_instr_acc,
            kl_loss_math=math_kl_val,
        )
        if len_pen_val is not None:
            wandb.log({"finetune/thinking_length_penalty": len_pen_val})
        _ifeval_str = (
            f"  ifeval={if_loss_val:.4f}  ifeval_prompt_acc={if_prompt_acc:.3f}"
            f"  ifeval_instr_acc={if_instr_acc:.3f}"
            if if_loss_val is not None else ""
        )
        _mkl_str = f"  kl_math={math_kl_val:.4f}" if math_kl_val is not None else ""
        _len_str = f"  len={len_pen_val:.4f}" if len_pen_val is not None else ""
        _pw_str = f"  probe_w={_det_w:g}" if _det_w != 1.0 else ""
        print(
            f"  step {step + 1:>4}/{cfg.finetune.n_steps} | "
            f"probe={p_loss:.4f}{_pw_str}  kl={k_loss:.4f}{_mkl_str}{_ifeval_str}{_len_str}"
            f"  total={t_loss:.4f}"
        )

        # ── Save checkpoint after every step ──────────────────────────────────
        step_ckpt_path = os.path.join(lora_all_ckpts_dir, f"step_{step + 1}")
        os.makedirs(step_ckpt_path, exist_ok=True)

        # The on-policy generations this step trained on, with their scores.
        if _onpol_rows:
            try:
                with open(os.path.join(step_ckpt_path, "onpolicy_generations.jsonl"), "w") as _f:
                    for _r in _onpol_rows:
                        _f.write(json.dumps(_r) + "\n")
            except Exception as e:  # noqa: BLE001
                print(f"    [safety] could not save on-policy generations: {e}")

        # Polytope diagnostics: geometry, membership and AUROC on the fixed set.
        if _diag is not None:
            try:
                _acts, _mask = _diag_activations(
                    _diag, model, tokenizer, polytope_layers, cfg, device)
                _pd = step_diagnostics(detector, polytope_layers, _acts,
                                       _diag["labels"], _mask, device=device)
                del _acts, _mask
                _pd["polytope/finetune_step"] = step + 1
                wandb.log(_pd)
                dump_step_diagnostics(_pd, step_ckpt_path)
                _k0 = f"polytope/L{polytope_layers[0]}/"
                print(f"    [polytope] eff_rank={_pd[_k0 + 'eff_rank']:.2f} "
                      f"dups={_pd[_k0 + 'dup_pairs']} "
                      f"viol/seq={_pd[_k0 + 'facets_violated_per_example']:.1f} "
                      f"harm_out={_pd[_k0 + 'frac_harmful_outside']:.3f} "
                      f"ben_out={_pd[_k0 + 'frac_benign_outside']:.3f} "
                      f"auroc={_pd[_k0 + 'auroc']:.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"    [polytope] diagnostics failed at step {step + 1}: {e}")
        save_lora_adapter(model, step_ckpt_path)
        save_probes(detector, os.path.join(step_ckpt_path, "probes"))

        # ── 5b. Held-out downstream eval for this step ────────────────────────
        # After the checkpoint is on disk: the vLLM subprocesses load the adapter
        # from it. Swap the HF model to CPU first so vLLM is not competing with a
        # resident 7B for the same GPU. The STOP-on-collapse decision is NOT made
        # here -- it comes from run_eval's utility further down.
        if _downstream is not None:
            try:
                model.to("cpu")
                torch.cuda.empty_cache()
                _downstream.run_step(step + 1)
            except Exception as e:  # noqa: BLE001
                print(f"    [downstream] step {step + 1} failed: {e}", flush=True)
            finally:
                model.to(device)
                model.train()

        # ── 6. Periodic probe retraining ──────────────────────────────────────
        is_retrain_step = (step + 1) % cfg.finetune.probe_retrain_interval == 0
        is_final_step = step == cfg.finetune.n_steps - 1
        # Frozen regime (mode=retrain, probe_retrain_steps=0) keeps the Step 0
        # probes unchanged for the whole run, so per-step retrain calls would
        # only re-extract activations and discard them. Skip entirely.
        _is_frozen = (
            cfg.finetune.get("probe_retrain_mode", "retrain") == "retrain"
            and cfg.finetune.get("probe_retrain_steps", 0) == 0
        )

        if (is_retrain_step or is_final_step) and not _is_frozen:
            freeze_model_params(model)
            model.eval()

            # Probe retrain sub-steps follow immediately after the finetune log step
            detector, wb_step, probe_step = retrain_probes(
                cfg, model, tokenizer, layers, detector, wb_step + 1, probe_step, step + 1
            )

            # Freeze probe params again before returning to model training
            for probe in detector.values():
                probe.requires_grad_(False)
                probe.eval()

            unfreeze_lora_params(model)
        elif _is_frozen and (is_retrain_step or is_final_step):
            print(f"  [step {step + 1}] frozen mode (probe_retrain_steps=0): "
                  f"skipping probe retraining (no activation extraction)")

        # ── 7. Periodic evaluation ────────────────────────────────────────────
        is_eval_step = (step + 1) % cfg.finetune.eval_interval == 0
        if is_eval_step or is_final_step:
            wb_step += 1
            print(f"\n  [step {global_step}] Running evaluation...")
            save_lora_adapter(model, lora_ckpt_path)
            model.eval()

            _step_utility_results = run_eval(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                detector=detector,
                layers=layers,
                lora_path=lora_ckpt_path,
                global_step=wb_step,
                finetune_step=step + 1,
            )
            model.train()

            # STOP-on-utility-collapse, from run_eval's own numbers rather than
            # a second utility pass or a sentinel file written by another job.
            if _utility_threshold is not None and _step_utility_results:
                _u = aggregate_utility(_step_utility_results)
                _stop_utility_collapsed = _u["aggregate"] < _utility_threshold
                print(f"  [utility] step {step + 1}: aggregate={_u['aggregate']:.4f} "
                      f"(threshold {_utility_threshold:.4f})"
                      + ("  << BELOW" if _stop_utility_collapsed else ""), flush=True)
                _ud = LIVE_DOWNSTREAM / wandb.run.id / "utility"
                _ud.mkdir(parents=True, exist_ok=True)
                (_ud / f"step_{step + 1}.json").write_text(json.dumps({
                    "step": step + 1,
                    "baseline": _baseline_utility,
                    "threshold": _utility_threshold,
                    "current": _u,
                    "below_threshold": bool(_stop_utility_collapsed),
                    # Which evaluator produced these numbers. aggregate_utility
                    # normalises to six keys, so the scorer is otherwise
                    # unrecoverable from the file and the plotting layer cannot
                    # tell an lm_eval in-loop run from an old hand-rolled one.
                    "scorer": _utility_scorer,
                }, indent=2))

        gc.collect()
        torch.cuda.empty_cache()

    # ── Save final adapter ────────────────────────────────────────────────────
    final_save = os.path.join(lora_all_ckpts_dir, "final_lora")
    save_lora_adapter(model, final_save)
    save_probes(detector, os.path.join(final_save, "probes"))
    print(f"\nFinal LoRA adapter saved to: {final_save}")

    # Sentinel for sibling watcher jobs: signals "no more steps coming".
    with open(os.path.join(lora_all_ckpts_dir, "FINETUNE_DONE"), "w") as _f:
        _f.write(f"completed n_steps={cfg.finetune.n_steps}\n")

    wandb.finish()


if __name__ == "__main__":
    finetune()
