"""DPO baseline, trained with trl, evaluated afterwards by the same code as the
probe/polytope runs.

Training saves a LoRA adapter per step and does no evaluation, so it runs at
full speed instead of pausing for a ~15 minute evaluation every step.

Utility and StrongREJECT are produced afterwards by src/probes/eval_checkpoint.py,
which uses the same functions as the finetuning loop and writes into the same
live_downstream directories, so the DPO and probe arms are evaluated identically.

Compute is matched to the probe/polytope arm on the levers that determine it:

    n_steps          50    same
    pairs per step   16    same as batch_size 8 harmful + 8 benign
    lr               5e-5  same
    LoRA             r=64, alpha=128, same target modules, same layers
    optimiser        AdamW, weight_decay 1e-4, no schedule

One honest difference remains and cannot be removed: DPO has no separable KL
term. beta is the coefficient of the KL in the RLHF objective DPO is derived
from, so it controls safety pressure and reference anchoring together, and there
is no DPO setting corresponding to "same detector strength, different lambda".
The comparison is therefore method-against-method, each at its own setting.

The reference policy is the same weights with the LoRA adapters disabled, which
is what trl does automatically when the model is a PeftModel and ref_model is
None -- the same trick compute_kl_loss uses, so no second copy of the model.

    DPO_CONFIG=/path/to/dpo_config.yaml WANDB_RUN_ID=abc12345 \
        python src/dpo_finetune.py
"""
import json
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
for _p in (SRC, os.path.join(SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                            # noqa: E402
import yaml                                             # noqa: E402

ROOT = os.environ.get("NM_ROOT", "/data/new_master")
LORA_ROOT = os.path.join(ROOT, "lora-finetuned")
CFG_PATH = os.environ["DPO_CONFIG"]
RUN_ID = os.environ["WANDB_RUN_ID"]
# Continue an existing run: load its step_<RESUME_STEP> adapter and train
# EXTRA_STEPS more, numbering the new checkpoints from RESUME_STEP+1. AdamW
# moments are not restored (trainer state was never saved), so the optimiser
# restarts cold at the boundary, the same warm-restart the finetune resume uses.
RESUME_STEP = int(os.environ.get("RESUME_STEP", "0"))
EXTRA_STEPS = int(os.environ.get("EXTRA_STEPS", "0"))


class SaveEveryStep:
    """Save the LoRA adapter after every optimizer step, as step_<N>.

    The probe/polytope runs write one checkpoint per step and everything
    downstream -- the evaluator, the plotting layer, the resume logic -- keys on
    that layout. Matching it exactly is what lets the DPO arm reuse all of it.
    """

    def __init__(self, out_dir, step_offset=0):
        self.out_dir = out_dir
        self.step_offset = step_offset

    def on_step_end(self, args, state, control, model=None, **kw):
        step = self.step_offset + int(state.global_step)
        d = os.path.join(self.out_dir, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        print(f"[dpo] saved {d}", flush=True)
        return control


def main():
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer
    from utils.finetune_utils import load_kl_rollouts, rollout_kl_backward

    cfg = yaml.safe_load(open(CFG_PATH))
    d = cfg["dpo"]
    model_name = cfg["model"]["model_name"]
    tok_name = cfg["model"].get("tokenizer_name") or model_name
    out_dir = os.path.join(LORA_ROOT, RUN_ID)
    os.makedirs(out_dir, exist_ok=True)

    # Same paired BeaverTails set the probe/polytope arm is fitted on, so the
    # two arms match on DATA and not only on compute. chosen = the safe
    # response, rejected = the unsafe one, both for the SAME prompt.
    from utils.jailbreak_datasets import build_beavertails_paired_probe_datasets
    pos, neg, _ = build_beavertails_paired_probe_datasets(
        n_train=int(d["n_train"]), n_eval=0)
    assert list(pos["prompt"]) == list(neg["prompt"]), "pairing broken"
    ds = Dataset.from_dict({
        "prompt": list(pos["prompt"]),
        "chosen": list(neg["completion"]),      # safe
        "rejected": list(pos["completion"]),    # unsafe
    })
    print(f"[dpo] {len(ds)} preference pairs from paired BeaverTails "
          f"(n_train={d['n_train']})", flush=True)

    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")

    lc = cfg["lora"]
    if RESUME_STEP > 0:
        from peft import PeftModel
        adapter = os.path.join(out_dir, f"step_{RESUME_STEP}")
        if not os.path.isdir(adapter):
            raise SystemExit(f"resume adapter not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        peft_config = None          # reuse the loaded adapter, do not re-init
        print(f"[dpo] resuming from {adapter}, training {EXTRA_STEPS} more steps "
              f"(checkpoints from step_{RESUME_STEP + 1})", flush=True)
    else:
        peft_config = LoraConfig(
            r=int(lc["r"]), lora_alpha=int(lc["lora_alpha"]),
            lora_dropout=float(lc.get("dropout", 0.0)),
            target_modules=list(lc["target_modules"]),
            task_type="CAUSAL_LM",
        )

    args = DPOConfig(
        output_dir=os.path.join(out_dir, "_trainer"),
        beta=float(d["beta"]),
        loss_type=d.get("loss_type", "sigmoid"),
        max_steps=(EXTRA_STEPS if RESUME_STEP > 0 else int(d["n_steps"])),
        per_device_train_batch_size=int(d["pairs_per_step"]),
        gradient_accumulation_steps=1,
        learning_rate=float(d["lr"]),
        weight_decay=float(d.get("weight_decay", 1e-4)),
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_length=int(d.get("max_length", 1024)),
        bf16=True,
        logging_steps=1,
        save_strategy="no",          # SaveEveryStep writes the layout we need
        report_to=[],
        remove_unused_columns=False,
        seed=42,
    )

    # Base order matters and is not cosmetic. TrainerCallback defines
    # on_step_end as a no-op, so with TrainerCallback first the MRO resolves
    # on_step_end to that stub and NOTHING is ever saved -- 50 steps of training
    # producing only a final adapter, with no error anywhere. SaveEveryStep must
    # come first.
    class _Cb(SaveEveryStep, TrainerCallback):
        pass


    # Optional explicit KL anchor: the SAME rollout-KL term the probe/polytope
    # runs use (same file, same coefficient, same rollout_kl_backward), added on
    # top of DPO's implicit (beta) KL. Lets the DPO arm be compared to the probe
    # arms with the KL regularisation controlled rather than confounded.
    kl_rollouts = []
    kcfg = {}
    if d.get("kl_rollouts"):
        kl_rollouts = load_kl_rollouts(d["kl_rollouts"],
                                       max_n=int(d.get("kl_max_pool", 1000)))
        kcfg = dict(penalty=float(d["kl_penalty"]),
                    batch_size_kl=int(d.get("batch_size_kl", 16)),
                    max_length=int(d.get("kl_max_seq_len_roll", 512)),
                    micro_batch=int(d.get("kl_micro_batch", 8)),
                    chunk_size=int(d.get("kl_chunk_size", 64)))
        print(f"[dpo] KL anchor: {len(kl_rollouts)} rollouts from {d['kl_rollouts']}"
              f", penalty={kcfg['penalty']}", flush=True)

    class DPOWithKL(DPOTrainer):
        """DPO plus the rollout-KL anchor. training_step runs the standard DPO
        backward, then rollout_kl_backward accumulates penalty*KL(base||current)
        over the frozen anchor completions into the same grads before the step."""

        def training_step(self, model, inputs, *a, **k):
            loss = super().training_step(model, inputs, *a, **k)
            import random as _r
            idx = _r.sample(range(len(kl_rollouts)),
                            min(kcfg["batch_size_kl"], len(kl_rollouts)))
            dev = next(self.model.parameters()).device
            rollout_kl_backward(
                self.model, [kl_rollouts[i] for i in idx], tok,
                penalty=kcfg["penalty"], max_length=kcfg["max_length"],
                micro_batch=kcfg["micro_batch"], chunk_size=kcfg["chunk_size"],
                device=dev)
            return loss

    TrainerCls = DPOWithKL if kl_rollouts else DPOTrainer
    trainer = TrainerCls(
        model=model,
        # ref_model=None with a PEFT model makes trl use the adapter-disabled
        # base as the reference, the same mechanism compute_kl_loss uses.
        ref_model=None,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=[_Cb(out_dir, step_offset=RESUME_STEP)],
    )
    print(f"[dpo] beta={d['beta']}  steps={d['n_steps']}  "
          f"pairs/step={d['pairs_per_step']}  lr={d['lr']}"
          + (f"  +KL(penalty={kcfg['penalty']})" if kl_rollouts else ""), flush=True)
    trainer.train()

    final = os.path.join(out_dir, "final_lora")
    trainer.model.save_pretrained(final)
    with open(os.path.join(out_dir, "FINETUNE_DONE"), "w") as f:
        f.write(json.dumps({"objective": "dpo", "config": CFG_PATH}))
    print(f"[dpo] done -> {final}", flush=True)


if __name__ == "__main__":
    main()
