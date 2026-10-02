#!/usr/bin/env python3
"""Grade MASK generations of a run, one snapshot per checkpoint step.

For each step: rebuild the six archetype response CSVs from
$NM_ROOT/live_downstream/<rid>/<sub>/step_<N>.json (build_step_csvs.py), grade
each with the LLM judge (evaluate_ds.py), aggregate with the official MASK
metric.py + process_metrics.py, and copy all_results.json to
<out-dir>/<rid>_step<NN>.json. Overall honesty is the response-weighted mean of
honesty_score_1 over the archetypes.

Idempotent: a step whose snapshot exists is skipped (unless --force). A step is
NOT snapshotted when the judge returned error/timeout sentinels, so a transient
API outage cannot masquerade as a (degenerate, honesty=100) result; rerun to
retry it.

The work dir holds the scratch csv_data/{responses,evaluated,metrics} tree and
grade.log; the official metric scripts use relative csv_data/ paths, so they are
copied into and run from the work dir.

Usage:
    export JUDGE_API_KEY=...      # optional: JUDGE_BASE_URL, JUDGE_MODEL
    python src/eval/mask/grade_mask.py --rid <rid> --steps 0-30 --work-dir mask_grading
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from build_step_csvs import NM_ROOT, build  # noqa: E402

TAG = "m"  # model tag in the CSV names; snapshots are keyed {archetype: {"m": {...}}}
SENTINEL = re.compile(r"TIMEOUT_ERROR|ERROR: Prompt")


def parse_steps(spec):
    """'0-30' / '0,5,10' / '0 5 10' -> sorted unique ints."""
    out = []
    for tok in spec:
        for part in tok.replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def overall_honesty(d):
    """Response-weighted mean of honesty_score_1 over archetypes -> (score, n)."""
    tot_n, tot = 0, 0.0
    for md in d.values():
        m = md.get(TAG, {})
        v, n = m.get("honesty_score_1"), m.get("total_responses", 0) or 0
        if v is not None:
            tot_n += n
            tot += v * n
    return (tot / tot_n if tot_n else None), tot_n


def _run(cmd, cwd, log, stdout_to_log=True):
    with open(log, "a") as fh:
        out = fh if stdout_to_log else subprocess.DEVNULL
        return subprocess.run(cmd, cwd=cwd, stdout=out, stderr=fh).returncode


def grade_step(rid, sub, st, work, out_dir, conc, nm_root, force):
    snap = os.path.join(out_dir, f"{rid}_step{st:02d}.json")
    if os.path.exists(snap) and not force:
        print(f"SKIP {rid} step {st} (have {snap})")
        return
    print(f"===== {rid} step {st} =====", flush=True)
    csv_root = os.path.join(work, "csv_data")
    for d in ("responses", "evaluated", "metrics"):
        os.makedirs(os.path.join(csv_root, d), exist_ok=True)
        for f in glob.glob(os.path.join(csv_root, d, "*.csv")):
            os.remove(f)
    stale = os.path.join(csv_root, "metrics", "all_results.json")
    if os.path.exists(stale):
        os.remove(stale)
    try:
        build(rid, sub, st, os.path.join(csv_root, "responses"), TAG, nm_root)
    except Exception as e:
        print(f"  build FAIL: {e}")
        return

    log = os.path.join(work, "grade.log")
    ok = True
    for f in sorted(glob.glob(os.path.join(csv_root, "responses", "*.csv"))):
        # Relative input path: evaluate_ds.py derives the output path by replacing
        # 'responses' -> 'evaluated' and picks the judge rules from the filename.
        rel = os.path.join("csv_data", "responses", os.path.basename(f))
        rc = _run([sys.executable, os.path.join(HERE, "evaluate_ds.py"),
                   "--input_file", rel, "--concurrency_limit", str(conc)], work, log)
        if rc != 0 or not os.path.exists(os.path.join(csv_root, "evaluated", os.path.basename(f))):
            print(f"  grade FAIL {rel}")
            ok = False
    if not ok:
        print(f"  grade FAIL {rid} step {st}: judge did not finish every archetype; not snapshotting")
        return
    # Guard: judge error/timeout sentinels would aggregate to a fake honesty=100.
    for f in glob.glob(os.path.join(csv_root, "evaluated", "*.csv")):
        with open(f, errors="replace") as fh:
            if SENTINEL.search(fh.read()):
                print(f"  grade FAIL (judge error/timeout sentinels) {rid} step {st}; not snapshotting")
                return

    for script in ("metric.py", "process_metrics.py"):
        if _run([sys.executable, script], work, log, stdout_to_log=False) != 0:
            print(f"  {script} FAIL {rid} step {st}; see {log}")
            return
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(stale, snap)
    score, n = overall_honesty(json.load(open(snap)))
    print(f"  -> overall honesty_score_1 = {score:.1f}  (n={n})" if n else "  -> (no data)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rid", required=True, help="finetune run id")
    ap.add_argument("--steps", nargs="+", required=True, help="e.g. 0-30 or 0 7 18")
    ap.add_argument("--work-dir", required=True, help="scratch dir (csv_data/, grade.log)")
    ap.add_argument("--out-dir", default=None, help="snapshot dir (default <work-dir>/curve_results)")
    ap.add_argument("--sub", default="mask", help="generation subdir under live_downstream/<rid>/")
    ap.add_argument("--nm-root", default=str(NM_ROOT), help="default: $NM_ROOT")
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("CONC", "16")))
    ap.add_argument("--force", action="store_true", help="regrade steps that already have a snapshot")
    args = ap.parse_args()

    if not os.environ.get("JUDGE_API_KEY"):
        raise SystemExit("ERROR: export JUDGE_API_KEY first (see evaluate_ds.py)")
    work = os.path.abspath(args.work_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(work, "curve_results"))
    os.makedirs(work, exist_ok=True)
    # metric.py resolves csv_data/ relative to its own location, so run copies.
    for script in ("metric.py", "process_metrics.py"):
        shutil.copy(os.path.join(HERE, script), os.path.join(work, script))

    for st in parse_steps(args.steps):
        grade_step(args.rid, args.sub, st, work, out_dir, args.concurrency, args.nm_root, args.force)
    print("CURVE_DONE")


if __name__ == "__main__":
    main()
