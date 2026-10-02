"""Merge SafeFlow outputs into one point list per n_train.

Utility comes from safeflow_eval.py outputs (per-lambda shards
safeflow_n<N>_lam<L>.json, then the combined safeflow_n<N>.json, which wins
ties), keyed by coef. StrongREJECT is then replaced by sr_mean from
safeflow_sr.py's safeflow_n<N>_lam<L>_sr.json wherever that exists.

steer_truncate is the cut the paper figures apply to every steering sweep: keep
points up to and including the first coef > 0 whose aggregate utility is below
util_floor times the coef 0 aggregate.

    python src/steering/merge_safeflow.py --n-train 750 --truncate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)


def load_safeflow_points(steering_dir, n):
    """Merged SafeFlow points for one n_train, sorted by coef."""
    d = str(steering_dir)
    shards = sorted(f for f in glob.glob(os.path.join(d, f"safeflow_n{n}_lam*.json"))
                    if not f.endswith("_sr.json"))
    combined = os.path.join(d, f"safeflow_n{n}.json")
    by_coef = {}
    for f in shards + ([combined] if os.path.exists(combined) else []):
        try:
            for p in json.load(open(f)).get("points", []):
                by_coef[p["coef"]] = dict(p)
        except (OSError, ValueError, KeyError):
            continue
    for f in glob.glob(os.path.join(d, f"safeflow_n{n}_lam*_sr.json")):
        try:
            s = json.load(open(f))
            c = s.get("lambda_unsafe")
            if c in by_coef and s.get("sr_mean") is not None:
                by_coef[c]["sr"] = s["sr_mean"]
        except (OSError, ValueError, KeyError):
            continue
    return [by_coef[k] for k in sorted(by_coef)]


def steer_truncate(points, util_floor=0.8):
    """Sort by coef and cut after the first coef > 0 point below the utility floor."""
    pts = sorted(points, key=lambda p: p["coef"])
    base = next((p["aggregate"] for p in pts if p["coef"] == 0), None)
    if not base:
        return pts
    floor = util_floor * base
    out = []
    for p in pts:
        out.append(p)
        if p["coef"] > 0 and p.get("aggregate") is not None and p["aggregate"] < floor:
            break
    return out


def main() -> None:
    from utils.paths import NM_ROOT

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--dir", default=None, help="default: $NM_ROOT/steering_eval")
    ap.add_argument("--truncate", action="store_true",
                    help="apply steer_truncate (as in the paper figures)")
    ap.add_argument("--util-floor", type=float, default=0.8)
    ap.add_argument("--out", default=None, help="write the merged points as JSON here")
    args = ap.parse_args()

    pts = load_safeflow_points(args.dir or (NM_ROOT / "steering_eval"), args.n_train)
    if args.truncate:
        pts = steer_truncate(pts, args.util_floor)
    print(f"{'lambda':>7} {'mmlu':>7} {'gsm8k':>7} {'ifeval':>7} {'agg':>7} {'sr':>7}")
    for p in pts:
        row = [p.get(k) for k in ("mmlu", "gsm8k", "ifeval_avg", "aggregate", "sr")]
        print(f"{p['coef']:>7g} " + " ".join(
            f"{v:7.4f}" if isinstance(v, (int, float)) else f"{'-':>7}" for v in row))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"method": "safeflow", "n_train": args.n_train,
                       "truncated": args.truncate, "points": pts}, fh, indent=2)


if __name__ == "__main__":
    main()
