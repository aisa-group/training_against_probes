"""SafeFlow StrongREJECT pass for one lambda_unsafe, saving every generation.

Same model, polytope injection and hook as safeflow_eval.py. Generates the 100
JailbreakBench completions at batch size 1, writes them after every prompt, then
releases the model and stores per-prompt StrongREJECT scores and their mean in
$NM_ROOT/steering_eval/safeflow_n<N>_lam<L>_sr.json. merge_safeflow.py takes SR
from this file and utility from safeflow_eval.py's shard.

lambda 0 runs without the hook (the unsteered base model).

    python src/steering/safeflow_sr.py --n-train 750 --polytope-ckpt <dir> --lambda 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SRC, "probes"))
sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from steering import _safeflow as sf  # noqa: E402
from utils.paths import NM_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sf.add_common_args(ap)
    ap.add_argument("--lambda", dest="lam", type=float, required=True,
                    help="lambda_unsafe; 0 = no steering (base model)")
    ap.add_argument("--smoke", action="store_true", help="8 JBB prompts")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: $NM_ROOT/steering_eval")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from asr.judges import score_strongreject_batch

    SafeRepModel = sf.import_safe_rep_model(args.safety_polytope_src)
    probe_dir, poly_label = sf.resolve_polytope(args)

    lam = args.lam
    sr_n = 8 if args.smoke else sf.SR_N
    stem = f"safeflow_n{args.n_train}_lam{lam:g}" + ("_smoke" if args.smoke else "")
    out_dir = args.out_dir or (NM_ROOT / "steering_eval")
    out = out_dir / f"{stem}_sr.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{stem}] building SafeRepModel (steer_layer={args.layer}, lambda={lam})",
          flush=True)
    safe, base, dev = sf.build_safeflow(SafeRepModel, args.model, args.layer, lam, probe_dir)
    print(f"[{stem}] injected polytope {probe_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    jbb, jbb_fmt = sf.load_jbb(tok, sr_n)

    handle = (base.model.layers[args.layer].register_forward_hook(sf.make_hook(safe))
              if lam != 0 else None)

    def _write(gens, scores=None):
        recs = []
        for i, (p, g) in enumerate(zip(jbb, gens)):
            r = {"idx": i, "prompt": p, "completion": g}
            if scores is not None:
                r["sr"] = float(scores[i])
            recs.append(r)
        out.write_text(json.dumps({
            "stem": stem, "method": "safeflow_sr", "n_train": args.n_train,
            "lambda_unsafe": lam, "base_model": args.model, "layer": args.layer,
            "polytope_run": poly_label, "polytope_ckpt": str(probe_dir),
            "num_iterations": (0 if lam == 0 else sf.NUM_ITERS),
            "sr_n": sr_n,
            "sr_mean": (float(np.mean(scores)) if scores is not None else None),
            "n_generated": len(gens), "generations": recs}, indent=2))

    def _progress(i, gens):
        print(f"[{stem}] generated {i + 1}/{sr_n}", flush=True)
        _write(gens)

    gens = sf.generate_one_by_one(base, tok, jbb_fmt, dev, on_step=_progress)

    if handle is not None:
        handle.remove()
    del safe, base
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{stem}] all {sr_n} generated; scoring StrongREJECT", flush=True)

    scores = score_strongreject_batch(jbb, gens)
    _write(gens, scores)
    print(f"[{stem}] SR mean = {float(np.mean(scores)):.4f}  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
