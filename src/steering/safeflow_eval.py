"""SafeFlow utility (and optionally StrongREJECT) over a lambda_unsafe sweep.

Runs the Safety Polytope paper's inference-time intervention (see _safeflow.py)
with our fitted K=16 polytope and scores the steered model at each lambda_unsafe:
lm-evaluation-harness utility (MMLU 5/subject, GSM8K and IFEval --util-limit
prompts, HF backend at batch size 1) and, unless --no-sr, StrongREJECT on 100
JailbreakBench prompts judged after the model is released.

The paper runs used one job per lambda with --shard-by-lambda, so each writes
$NM_ROOT/steering_eval/safeflow_n<N>_lam<L>.json. Lambda 0 was run with SR here;
lambdas > 0 were run with --no-sr, and their SR comes from safeflow_sr.py.
merge_safeflow.py combines the files.

    python src/steering/safeflow_eval.py --n-train 750 --polytope-ckpt <dir> \\
        --shard-by-lambda --lambdas 2 --no-sr
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

LAMBDAS = (0.0, 2.0, 4.0, 10.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sf.add_common_args(ap)
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS),
                    help="lambda_unsafe values; 0 means no unsafe-violation pressure")
    ap.add_argument("--smoke", action="store_true",
                    help="first 2 lambdas, tiny utility limits, 8 JBB prompts")
    ap.add_argument("--util-limit", type=int, default=100,
                    help="GSM8K and IFEval prompt cap (MMLU stays 5 per subject)")
    ap.add_argument("--shard-by-lambda", action="store_true",
                    help="write safeflow_n<N>_lam<L>.json for a single --lambdas value "
                         "instead of the combined safeflow_n<N>.json")
    ap.add_argument("--no-sr", action="store_true",
                    help="utility only (sr=None); SR comes from safeflow_sr.py")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: $NM_ROOT/steering_eval")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from asr.downstream_eval import aggregate_utility
    from asr.judges import score_strongreject_batch
    from utils.lmeval_utils import harvest, task_manager, tasks  # before lm_eval: sets up LMEVAL_PKGS
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    SafeRepModel = sf.import_safe_rep_model(args.safety_polytope_src)
    probe_dir, poly_label = sf.resolve_polytope(args)

    lambdas = args.lambdas[:2] if args.smoke else args.lambdas
    mmlu_lim, gsm_lim, if_lim = ((2, 5, 5) if args.smoke
                                 else (5, args.util_limit, args.util_limit))
    sr_n = 8 if args.smoke else sf.SR_N
    if args.shard_by_lambda:
        if len(lambdas) != 1:
            raise SystemExit("--shard-by-lambda requires exactly one --lambdas value")
        stem = f"safeflow_n{args.n_train}_lam{lambdas[0]:g}" + ("_smoke" if args.smoke else "")
    else:
        stem = f"safeflow_n{args.n_train}" + ("_smoke" if args.smoke else "")
    out_dir = args.out_dir or (NM_ROOT / "steering_eval")
    out = out_dir / f"{stem}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{stem}] building SafeRepModel (steer_layer={args.layer})", flush=True)
    safe, base, dev = sf.build_safeflow(SafeRepModel, args.model, args.layer,
                                        lambdas[0], probe_dir)
    print(f"[{stem}] injected polytope {probe_dir}: phi {tuple(safe.phi.shape)}, "
          f"threshold {tuple(safe.threshold.shape)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    jbb, jbb_fmt = sf.load_jbb(tok, sr_n)

    handle = base.model.layers[args.layer].register_forward_hook(sf.make_hook(safe))

    def _utility():
        """MMLU / GSM8K / IFEval with the hook active (MMLU without chat template)."""
        lm = HFLM(pretrained=base, tokenizer=args.model, batch_size=1, max_length=4096)
        out_d = {}
        for task, limit in tasks(mmlu_lim, gsm_lim, if_lim):
            res = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, limit=limit,
                                  apply_chat_template=(task != "mmlu"),
                                  task_manager=task_manager())
            harvest(res, out_d)
        return aggregate_utility(out_d)

    def _write(points):
        out.write_text(json.dumps({
            "stem": stem, "method": "safeflow", "n_train": args.n_train,
            "base_model": args.model, "layer": args.layer, "polytope_run": poly_label,
            "polytope_ckpt": str(probe_dir),
            "num_iterations": sf.NUM_ITERS, "lambda_safe": sf.LAMBDA_SAFE,
            "scorer": "lm_eval", "sr_n": sr_n, "util_limit": gsm_lim,
            "points": points}, indent=2))

    swept = []                  # (lambda_unsafe, utility, generations)
    for lam in lambdas:
        safe.lambda_weight = lam       # read by optimize_hidden_states
        u = _utility()
        g = None if args.no_sr else sf.generate_one_by_one(base, tok, jbb_fmt, dev)
        swept.append((lam, u, g))
        print(f"[{stem}] lambda_unsafe={lam:.2f}  "
              f"util(mmlu/gsm8k/ifeval)={u['mmlu']:.3f}/{u['gsm8k']:.3f}/{u['ifeval_avg']:.3f}"
              f"  agg={u['aggregate']:.3f}  (SR pending)", flush=True)
        _write([{"coef": a, "mmlu": uu["mmlu"], "gsm8k": uu["gsm8k"],
                 "ifeval_avg": uu["ifeval_avg"], "aggregate": uu["aggregate"], "sr": None}
                for a, uu, _g in swept])

    if args.no_sr:
        print(f"[{stem}] utility only (--no-sr): wrote {out}", flush=True)
        return

    # Release the model before the judge's vLLM process starts.
    handle.remove()
    del safe, base
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{stem}] utility+generation done; judging StrongREJECT", flush=True)

    points = []
    for lam, u, g in swept:
        sr = float(np.mean(score_strongreject_batch(jbb, g)))
        points.append({"coef": lam, "mmlu": u["mmlu"], "gsm8k": u["gsm8k"],
                       "ifeval_avg": u["ifeval_avg"], "aggregate": u["aggregate"], "sr": sr})
        print(f"[{stem}] lambda_unsafe={lam:.2f}  SR={sr:.3f}", flush=True)
        _write(points)
    print(f"[{stem}] wrote {out} ({len(points)} points)", flush=True)


if __name__ == "__main__":
    main()
