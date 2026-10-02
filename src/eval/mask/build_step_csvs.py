#!/usr/bin/env python3
"""Reconstruct the 6 MASK archetype responses-CSVs for ONE checkpoint step from
its stored per-step generation json ($NM_ROOT/live_downstream/<rid>/<sub>/step_<n>.json,
written by src/probes/mask_generate.py), in the exact column layout the judge
(evaluate_ds.py) expects.

Usage:
    python src/eval/mask/build_step_csvs.py <rid> <sub> <step> <out_dir> <tag> [--nm-root DIR]

Writes <out_dir>/<archetype>_<tag>.csv (tag must contain no underscore, since the
metric scripts parse archetype/model by splitting on the last underscore).
"""
import argparse
import csv
import json
import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.paths import NM_ROOT  # noqa: E402  (env NM_ROOT, default /data/new_master)

BASE_COLS = [
    "proposition", "formatted_ground_truth", "ground_truth", "type",
    "system_prompt", "user_prompt",
    "belief_elicit_1", "belief_elicit_2", "belief_elicit_3", "belief_system_1",
]
# Extra metadata columns for doubling_down, inserted before the generation cols.
DD_COLS = ["dd_turn_1_prompt", "dd_turn_1_response"]

# Generation columns per archetype (match the canonical responses CSV headers).
GEN_COLS = {
    "continuations": [
        "generation(System Prompt + User Prompt)_run1",
        "generation(belief_system_1 + belief_elicit_1)_run1",
        "generation(belief_system_1 + belief_elicit_1)_run2",
        "generation(belief_system_1 + belief_elicit_1)_run3",
        "generation(belief_system_1 + belief_elicit_2)_run1",
        "generation(belief_system_1 + belief_elicit_3)_run1",
    ],
    "provided_facts": [
        "generation(System Prompt + User Prompt)_run1",
    ],
    "statistics": [
        "generation(System Prompt + User Prompt)_run1",
        "generation(belief_system_1 + belief_elicit_1)_run1",
        "generation(belief_system_1 + belief_elicit_1)_run2",
        "generation(belief_system_1 + belief_elicit_1)_run3",
    ],
}
# disinformation / known_facts / doubling_down share the continuations gen cols.
for a in ("disinformation", "known_facts", "doubling_down_known_facts"):
    GEN_COLS[a] = GEN_COLS["continuations"]


def build(rid, sub, step, out_dir, tag, nm_root=NM_ROOT):
    """Write <out_dir>/<archetype>_<tag>.csv for every archetype in one step file."""
    assert "_" not in tag, "tag must not contain '_'"
    src = os.path.join(str(nm_root), "live_downstream", rid, sub, f"step_{step}.json")
    d = json.load(open(src))
    os.makedirs(out_dir, exist_ok=True)

    by_arch = {}
    for e in d["examples"]:
        by_arch.setdefault(e["archetype"], []).append(e)

    for arch, rows in by_arch.items():
        gen_cols = GEN_COLS[arch]
        cols = list(BASE_COLS)
        if arch == "doubling_down_known_facts":
            cols += DD_COLS
        cols += gen_cols
        path = os.path.join(out_dir, f"{arch}_{tag}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for e in rows:
                oc = e.get("official_columns", {})
                row = {c: e.get(c, "") for c in BASE_COLS}
                if arch == "doubling_down_known_facts":
                    for c in DD_COLS:
                        row[c] = e.get(c, "")
                for c in gen_cols:
                    row[c] = oc.get(c, "")
                w.writerow(row)
        print(f"  wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rid")
    ap.add_argument("sub", help="generation subdir, e.g. mask")
    ap.add_argument("step")
    ap.add_argument("out_dir")
    ap.add_argument("tag", help="model tag in the CSV names (no underscore)")
    ap.add_argument("--nm-root", default=str(NM_ROOT), help="default: $NM_ROOT")
    a = ap.parse_args()
    build(a.rid, a.sub, a.step, a.out_dir, a.tag, a.nm_root)


if __name__ == "__main__":
    main()
