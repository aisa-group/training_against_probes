"""Shared SafeFlow setup for safeflow_eval.py and safeflow_sr.py.

SafeFlow is the inference-time intervention of the Safety Polytope paper (Chen,
As, Krause 2025). Only SafeRepModel.check_constraint and
SafeRepModel.optimize_hidden_states from the authors' code are used. Their
steer_forward is replaced by a forward hook on the steered decoder layer that
applies the same intervention to the last position of every forward pass: if
the hidden state lies outside the polytope, it is replaced by the result of 100
SGD steps (lr 0.01) on ||dh||_1 / d + lambda_safe * sum(viol)
+ lambda_unsafe * sum(relu(viol)). Generation runs at batch size 1 because
optimize_hidden_states assumes a single steered position.

The polytope is the K=16 PolytopeProbe fitted on the base model (the step-1
detector of a polytope-guided run). Its feature_extractor, phi and threshold map
directly onto SafeRepModel's attributes of the same names.

SafetyPolytope source lookup order: --safety-polytope-src, $SAFETY_POLYTOPE_SRC,
then third_party/SafetyPolytope/src (see third_party/fetch_safety_polytope.sh).
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import torch

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LAYER = 19
SR_N = 100
SR_MAX_NEW = 256
NUM_ITERS = 100               # paper Appendix B; the authors' call defaults to 1
STEER_ALL_TOKENS = 100_000    # steer every generated token (authors' default caps at 20)
LAMBDA_SAFE = 1e-4            # authors' default safe_violation_weight

SAFETY_POLYTOPE_COMMIT = "137096bb9f683842ff0f58754e980cb8e3824bd5"
_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_SP_SRC = _REPO / "third_party" / "SafetyPolytope" / "src"


def add_common_args(ap) -> None:
    """CLI options shared by the SafeFlow scripts."""
    ap.add_argument("--n-train", type=int, default=750,
                    help="paired BeaverTails size the polytope was fitted on (output naming)")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--layer", type=int, default=LAYER)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--polytope-ckpt", type=Path,
                   help="PolytopeProbe directory (config.json + model.pt), "
                        "e.g. <run>/step_1/probes/layer_19")
    g.add_argument("--rid", help="run id under $NM_ROOT/lora-finetuned")
    ap.add_argument("--step", type=int, default=1,
                    help="checkpoint step used with --rid (default 1: the base-model detector)")
    ap.add_argument("--safety-polytope-src", type=Path, default=None,
                    help="SafetyPolytope src/ dir (default: $SAFETY_POLYTOPE_SRC or "
                         "third_party/SafetyPolytope/src)")


def import_safe_rep_model(src: Path | None = None):
    """Put the SafetyPolytope src/ dir on sys.path and return SafeRepModel."""
    src = Path(src or os.environ.get("SAFETY_POLYTOPE_SRC") or _DEFAULT_SP_SRC)
    if not (src / "safety_polytope" / "polytope" / "safe_rep_model.py").is_file():
        raise SystemExit(
            f"SafetyPolytope not found at {src}. Run third_party/fetch_safety_polytope.sh "
            f"or set SAFETY_POLYTOPE_SRC to the src/ dir of a checkout of "
            f"github.com/lasgroup/SafetyPolytope at {SAFETY_POLYTOPE_COMMIT}.")
    sys.path.insert(0, str(src))
    from safety_polytope.polytope.safe_rep_model import SafeRepModel
    return SafeRepModel


def resolve_polytope(args) -> tuple[Path, str | None]:
    """Return (probe dir, run label) from --polytope-ckpt or --rid/--step."""
    if args.rid:
        from utils.paths import LORA_ROOT
        path = LORA_ROOT / args.rid / f"step_{args.step}" / "probes" / f"layer_{args.layer}"
        label = args.rid
    else:
        path = args.polytope_ckpt
        # <run>/step_<s>/probes/layer_<L>: label with the run directory name.
        p = path.resolve().parts
        label = p[-4] if len(p) >= 4 and p[-2] == "probes" and p[-3].startswith("step_") else None
    if not (path / "config.json").is_file():
        raise SystemExit(f"no probe checkpoint (config.json) at {path}")
    return path, label


def build_safeflow(SafeRepModel, model_name, layer, lam, probe_dir):
    """Load the base model inside SafeRepModel and inject the fitted polytope.

    Returns (safe, base, device): the SafeRepModel, its plain causal LM, and the
    device of its first parameter (where generation inputs go).
    """
    from train import load_model as load_probe
    import probe_archs  # noqa: F401  (PolytopeProbe class referenced by config.json)

    safe = SafeRepModel(
        model_name, steer_layer=layer, steer_first_n_tokens=STEER_ALL_TOKENS,
        lambda_weight=lam, safe_violation_weight=LAMBDA_SAFE,
        use_backup_response=False, projection=False,
        torch_dtype=torch.bfloat16, device_map="cuda")
    safe.eval()
    base = safe.lm_model
    base.eval()

    probe = load_probe(str(probe_dir)).eval()
    dev = next(safe.model.parameters()).device
    dt = next(safe.model.parameters()).dtype
    safe.feature_extractor = probe.feature_extractor.to(device=dev, dtype=dt)
    safe.phi = torch.nn.Parameter(probe.phi.detach().to(device=dev, dtype=dt))
    safe.threshold = torch.nn.Parameter(probe.threshold.detach().to(device=dev, dtype=dt))
    return safe, base, dev


def make_hook(safe, num_iters=NUM_ITERS):
    """Forward hook applying the SafeFlow projection to the last position.

    lambda_unsafe is read from safe.lambda_weight at call time.
    """
    def _safeflow_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out          # (B, S, D)
        last = h[:, -1, :]
        feats = safe.feature_extractor(last)
        unsafe = ~safe.check_constraint(feats)                 # True = outside polytope
        if bool(unsafe.any()):
            with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                opt = safe.optimize_hidden_states(
                    last, unsafe, num_iterations=num_iters, verbose=False)
            h = h.clone()
            h[:, -1, :] = opt.to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return out
    return _safeflow_hook


def load_jbb(tok, n):
    """First n JailbreakBench prompts and their chat-formatted versions."""
    from utils.finetune_utils import apply_chat_template
    from utils.jailbreak_datasets import load_jailbreakbench
    jbb = load_jailbreakbench()[:n]
    return jbb, apply_chat_template(jbb, tok)


def generate_one_by_one(base, tok, prompts_fmt, dev, on_step=None):
    """Greedy SR_MAX_NEW-token completions at batch size 1."""
    gen = []
    for i, p in enumerate(prompts_fmt):
        enc = tok([p], return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            o = base.generate(**enc, max_new_tokens=SR_MAX_NEW, do_sample=False,
                              pad_token_id=tok.pad_token_id)
        gen.append(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if on_step is not None:
            on_step(i, gen)
    return gen
