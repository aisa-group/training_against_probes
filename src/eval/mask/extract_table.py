#!/usr/bin/env python3
"""Per-archetype MASK honesty table (+ response-weighted Average) from grade_mask.py
snapshots, with deltas vs. the first point and LaTeX rows (fractions, 2 dp).

Usage:
    python src/eval/mask/extract_table.py --results-dir mask_grading/curve_results \\
        --point base=<rid>_step00 --point "probe (s7)=<rid>_step07" \\
        --point "poly K16 (s18)=<rid2>_step18"
The first --point is the baseline for the deltas.
"""
import argparse
import json
import os

COLS = ["continuations", "disinformation", "doubling_down_known_facts",
        "known_facts", "provided_facts", "statistics"]
PRETTY = {"continuations": "Continuations", "disinformation": "Disinformation",
          "doubling_down_known_facts": "Doubling down", "known_facts": "Known facts",
          "provided_facts": "Provided facts", "statistics": "Statistics"}


def load(results_dir, snap):
    d = json.load(open(os.path.join(results_dir, snap + ".json")))
    vals, ns = {}, {}
    for a in COLS:
        m = d.get(a, {}).get("m", {})
        vals[a] = m.get("honesty_score_1")
        ns[a] = m.get("total_responses", 0) or 0
    tot_n = sum(ns.values())
    tot = sum((vals[a] or 0) * ns[a] for a in COLS)
    vals["Average"] = round(tot / tot_n, 3) if tot_n else None
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="grade_mask.py --out-dir")
    ap.add_argument("--point", action="append", required=True,
                    help="LABEL=SNAPSHOT (snapshot name without .json, e.g. base=<rid>_step00)")
    args = ap.parse_args()
    points = [tuple(p.split("=", 1)) for p in args.point]
    CR = args.results_dir

    table = {}
    for lab, snap in points:
        p = os.path.join(CR, snap + ".json")
        if not os.path.exists(p):
            print(f"MISSING {snap} -> cannot build table yet"); return
        table[lab] = load(CR, snap)
    hdr = [PRETTY[c] for c in COLS] + ["Average"]
    print(f"{'model':16s} " + " ".join(f"{h[:13]:>13s}" for h in hdr))
    for lab, _ in points:
        v = table[lab]
        cells = [(v[c]) for c in COLS] + [v["Average"]]
        print(f"{lab:16s} " + " ".join((f"{x/100:>13.2f}" if x is not None else f"{'n/a':>13s}") for x in cells))
    # deltas vs base (fraction points)
    print("\n-- deltas vs base (fraction) --")
    b = table[points[0][0]]
    for lab, _ in points[1:]:
        v = table[lab]
        cells = [(v[c] - b[c]) / 100 if v[c] is not None and b[c] is not None else None for c in COLS]
        avg = (v["Average"] - b["Average"]) / 100 if v["Average"] is not None else None
        print(f"{lab:16s} " + " ".join((f"{x:+13.2f}" if x is not None else f"{'n/a':>13s}") for x in cells + [avg]))
    # LaTeX fractions (2dp) for quick paste
    print("\n-- LaTeX (values only, 2dp) --")
    for lab, _ in points:
        v = table[lab]
        cells = [v[c] for c in COLS] + [v["Average"]]
        print(f"% {lab}: " + " & ".join((f"{x/100:.2f}" if x is not None else "--") for x in cells))


if __name__ == "__main__":
    main()
