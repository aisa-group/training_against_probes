from __future__ import annotations

import os

import wandb


WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
# WANDB_PROJECT selects the experiment tracking project.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "polytope-training")


def init_run(
    config: dict,
    name: str | None = None,
    tags: list[str] | None = None,
    id: str | None = None,
    resume: str = "allow",
) -> wandb.sdk.wandb_run.Run:
    """Initialise a wandb run.

    Args:
        config: Hyperparameters and settings to record.
        name: Optional human-readable run name.
        tags: Optional list of tags.
        id: Optional pre-generated run id. When provided, the run is created
            with this exact id, allowing separate processes to log to the same
            run via ``resume="allow"``.
        resume: Resume policy when ``id`` is provided. Defaults to "allow".

    Returns:
        The newly created wandb run (also accessible via ``wandb.run``).
    """
    kwargs = dict(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        dir="/data/new_master/wandb",
        config=config,
        name=name,
        tags=tags,
    )
    if id is not None:
        kwargs["id"] = id
        kwargs["resume"] = resume
    run = wandb.init(**kwargs)
    # Track probe-fitting metrics against probe_step. Keep the step key outside
    # the probe_train/* wildcard to avoid applying the rule to the counter.
    wandb.define_metric("probe_train/*", step_metric="probe_step")
    # Track model-update metrics against finetune_step independently of probe-fitting steps.
    wandb.define_metric("utility/*", step_metric="utility_step")
    # Custom step axis for the finetune-level timeline (probe_loss, KL loss,
    # eval probe AUC, etc.). Using an explicit step= in wandb.log breaks on
    # resume — when finetune.py restarts and tries to log a small step number,
    # wandb rejects it as "less than current step". Routing everything through
    # custom step metrics + auto-incremented internal _step sidesteps that.
    wandb.define_metric("finetune_step")
    wandb.define_metric("finetune/*", step_metric="finetune_step")
    wandb.define_metric("eval/*", step_metric="finetune_step")
    wandb.define_metric("probe_eval/*", step_metric="finetune_step")
    return run


def log_probe_train_dynamics(
    train_dynamics: dict,
    global_step_offset: int = 0,
    probe_step_offset: int = 0,
    finetune_step: int = 0,
) -> tuple[int, int]:
    """Log per-step probe training curves to the active wandb run.

    Metrics are logged under the ``probe_train/`` prefix, e.g. ``probe_train/loss``,
    ``probe_train/lr``, ``probe_train/layer_5_loss``.

    ``probe_step`` is included in every log dict so that W&B uses it as
    the x-axis for all ``probe_train/*`` charts (via ``define_metric``).  This
    makes the learning curve continuous across the initial probe training and
    all subsequent retraining phases.  The name is deliberately kept outside
    the ``probe_train/*`` wildcard to avoid a circular dependency.

    Args:
        train_dynamics: Dict returned by ``train_and_eval_detector`` with keys
            like ``"layer_X_loss"``, ``"loss"``, ``"lr"`` and list-of-float
            values (one entry per optimiser step).
        global_step_offset: Added to the local step index for the global W&B
            timeline step (keeps the global run step monotonically increasing).
        probe_step_offset: Cumulative number of probe training steps already
            logged in earlier phases.  Added to the local step index so the
            probe learning curve is continuous across phases.
        finetune_step: Which finetuning iteration triggered this probe training
            (0 = initial training before any finetuning).  Logged as
            ``probe_train/finetune_step`` at every sub-step so it appears as a
            staircase that lets you map probe_step ↔ finetune_step.

    Returns:
        (last_global_step, last_probe_step): The last values of each counter,
        or (global_step_offset - 1, probe_step_offset - 1) if nothing logged.
    """
    if not train_dynamics or wandb.run is None:
        return global_step_offset - 1, probe_step_offset - 1
    n_steps = max((len(v) for v in train_dynamics.values() if isinstance(v, list)), default=0)
    for local_step in range(n_steps):
        log_dict: dict[str, float] = {
            "probe_step": probe_step_offset + local_step,
            "probe_train/finetune_step": finetune_step,
        }
        for key, values in train_dynamics.items():
            if not (isinstance(values, list) and local_step < len(values)):
                continue
            if key == "lr":
                log_dict["probe_train/lr"] = values[local_step]
            elif key == "loss":
                log_dict["probe_train/loss"] = values[local_step]
            elif key.startswith("layer_") and key.endswith("_loss"):
                layer = key[len("layer_") : -len("_loss")]
                log_dict[f"probe_train/layer_{layer}_loss"] = values[local_step]
        # Use probe_step as the metric x-axis; let wandb auto-increment _step
        # (avoids "step N < current step" rejection on resume).
        wandb.log(log_dict)
    if n_steps > 0:
        return global_step_offset + n_steps - 1, probe_step_offset + n_steps - 1
    return global_step_offset - 1, probe_step_offset - 1


def log_finetune_step(
    step: int,
    probe_loss: float,
    kl_loss: float,
    total_loss: float,
    ifeval_loss: float | None = None,
    ifeval_prompt_accuracy: float | None = None,
    ifeval_instruction_accuracy: float | None = None,
    kl_loss_math: float | None = None,
) -> None:
    """Log per-step finetuning losses.

    Args:
        step: Global optimizer step (used as W&B x-axis).
        probe_loss: Normalized probe BCE loss.
        kl_loss: Mean token-level KL divergence.
        total_loss: Weighted sum of all active loss terms.
        ifeval_loss: CE loss on passing IFEval completions (None if disabled).
        ifeval_prompt_accuracy: Fraction of IFEval prompts where all instructions passed.
        ifeval_instruction_accuracy: Fraction of individual IFEval instructions passed.
        kl_loss_math: Mean KL of the maths anchor (None if no maths anchor).
    """
    if wandb.run is None:
        return
    log_dict: dict[str, float] = {
        "finetune_step": step,
        "finetune/probe_loss": probe_loss,
        "finetune/kl_loss": kl_loss,
        "finetune/total_loss": total_loss,
    }
    if kl_loss_math is not None:
        log_dict["finetune/kl_loss_math"] = kl_loss_math
    if ifeval_loss is not None:
        log_dict["finetune/ifeval_loss"] = ifeval_loss
    if ifeval_prompt_accuracy is not None:
        log_dict["finetune/ifeval_prompt_accuracy"] = ifeval_prompt_accuracy
    if ifeval_instruction_accuracy is not None:
        log_dict["finetune/ifeval_instruction_accuracy"] = ifeval_instruction_accuracy
    wandb.log(log_dict)


def log_finetune_eval(
    step: int,
    jailbreak_rates: dict[str, float],
    probe_auc_roc: dict[int, float],
    probe_test_auc_roc: dict[str, dict] | None = None,
) -> None:
    """Log finetuning evaluation metrics.

    Args:
        step: Global step (x-axis).
        jailbreak_rates: {split_name: rate} from WildGuard.
        probe_auc_roc: {layer_idx: auc_roc} from on-policy activations + WildGuard labels.
        probe_test_auc_roc: Optional {pair_name: {layer: metrics}} from static test sets.
    """
    if wandb.run is None:
        return
    log_dict: dict[str, float] = {"finetune_step": step}

    for split, rate in jailbreak_rates.items():
        log_dict[f"eval/jailbreak_rate/{split}"] = rate

    for layer, auc in probe_auc_roc.items():
        log_dict[f"eval/probe_auc_roc/layer_{layer}"] = auc

    if probe_test_auc_roc is not None:
        for pair_name, layer_metrics in probe_test_auc_roc.items():
            for layer, metrics in layer_metrics.items():
                auc = metrics.get("AUC_ROC", float("nan"))
                if isinstance(auc, (int, float)):
                    log_dict[f"probe_eval/{pair_name}/AUC_ROC/layer_{layer}"] = auc

    wandb.log(log_dict)
    wandb.run.summary.update(log_dict)


def log_utility_eval(step: int, results: dict[str, float], finetune_step: int) -> None:
    """Log utility evaluation metrics to the active W&B run.

    All keys in *results* should be prefixed with ``utility/``.  Every value
    is written both to the step timeline and to ``wandb.run.summary`` so it
    appears in the runs table. ``utility_step`` is included so charts can
    plot against finetune iteration (see ``define_metric`` in ``init_run``).

    Args:
        step: Global W&B step (must be monotonically increasing across the run).
        results: Flat dict of metric_name → float, e.g.
            {"utility/mmlu_accuracy": 0.62, "utility/ppl_ratio": 1.03, ...}.
        finetune_step: Finetune iteration (0 = baseline). Logged as
            ``utility_step`` so ``utility/*`` charts use it as x-axis.
    """
    if wandb.run is None or not results:
        return
    log_dict = {**results, "utility_step": finetune_step}
    wandb.log(log_dict)
    wandb.run.summary.update(results)


def log_probe_eval_metrics(results: dict[str, dict], step: int | None = None) -> None:
    """Log AUC-ROC per eval-pair and layer to the active wandb run.

    Metrics are logged under ``probe_eval/<pair>/AUC_ROC/layer_<N>`` and also
    written to ``wandb.run.summary`` for the table view.

    All metrics are batched into a single ``wandb.log`` call so that the W&B
    step counter is not auto-incremented once per metric.

    Args:
        results: Dict keyed by eval-pair name → layer → metric dict, as
            returned by ``train_and_evaluate`` in ``compare_probes.py``.
        step: Explicit W&B step.  Must be provided (and monotonically
            increasing) when this function is part of a run that also uses
            explicit steps elsewhere; pass ``None`` only for standalone runs.
    """
    if not results or wandb.run is None:
        return
    log_dict: dict[str, float] = {}
    for pair_name, layer_metrics in results.items():
        for layer, metrics in layer_metrics.items():
            auc_roc = metrics.get("AUC_ROC")
            if isinstance(auc_roc, (int, float)):
                key = f"probe_eval/{pair_name}/AUC_ROC/layer_{layer}"
                log_dict[key] = auc_roc
    if log_dict:
        if step is not None:
            log_dict["finetune_step"] = step
        wandb.log(log_dict)
        wandb.run.summary.update(log_dict)
