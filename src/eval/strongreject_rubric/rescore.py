"""Re-judge ClearHarm continuations with the StrongREJECT rubric judge.

Takes result files written by src/attacks/clearharm_direct_prefill.py (direct, prefill)
and src/attacks/gcg_regenerate.py (GCG), or directories containing them, and
writes deepseek_<name>.json next to each input (step_N.json ->
deepseek_step_N.json, base.json -> deepseek_base.json). No generation is done.
The judged behavior is always the original ClearHarm query, never the attacked
text. Items whose API call or parse fails get score None and are excluded from
mean/sd. Existing outputs are kept unless --force.

    JUDGE_API_KEY=... python src/eval/strongreject_rubric/rescore.py \\
        $NM_ROOT/live_downstream/<rid> $NM_ROOT/clearharm_sr/<name>
"""
import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubric_judge import MODEL, score_rubric  # noqa: E402

PREFIX = "deepseek_"
_INPUT_NAME = re.compile(r"^(step_\d+|base)\.json$")


def find_inputs(paths):
    """Expand files/dirs into result files. Directories are walked recursively
    and only ClearHarm result files (step_N.json / base.json with
    "dataset": "ClearHarm") are kept; explicit files are taken as given."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
            continue
        if not os.path.isdir(p):
            raise SystemExit(f"[rescore] not found: {p}")
        for root, _, files in os.walk(p):
            for f in sorted(files):
                if not _INPUT_NAME.match(f):
                    continue
                fp = os.path.join(root, f)
                try:
                    with open(fp) as fh:
                        if json.load(fh).get("dataset") != "ClearHarm":
                            continue
                except (OSError, ValueError):
                    continue
                out.append(fp)
    return sorted(dict.fromkeys(out))


def output_path(inf):
    d, f = os.path.split(inf)
    return os.path.join(d, PREFIX + f)


def _beh(r):
    # GCG files use query/response; direct/prefill use prompt/response.
    return r.get("prompt") or r.get("query") or ""


def _gen(r):
    return r.get("completion") or r.get("response") or ""


def _input_hash(pairs):
    """Fingerprint of the judged (behavior, generation) pairs."""
    h = hashlib.sha256()
    for b, g in pairs:
        h.update(json.dumps([b, g]).encode())
    return h.hexdigest()


def _cached_mean(outf, pairs, digest):
    """Mean of an existing judgement if it was made on exactly these inputs, else None.

    Judgements carry the hash of their inputs; older files without one are
    accepted only when they cover the same prompts in the same order."""
    try:
        with open(outf) as fh:
            prev = json.load(fh)
    except (OSError, ValueError):
        return None
    if "input_sha256" in prev:
        ok = prev["input_sha256"] == digest
    else:
        ok = (prev.get("n_total") == len(pairs)
              and [r.get("prompt") for r in prev.get("results", [])] == [b for b, _ in pairs])
    return prev.get("mean_p_harmful") if ok else None


def rescore(inf, outf, force=False):
    """Judge one result file; returns the mean over successfully scored items.

    An existing judgement is reused only if it was made on the current inputs, so
    regenerated result files are always judged again."""
    with open(inf) as fh:
        res = json.load(fh)["results"]
    pairs = [(_beh(r), _gen(r)) for r in res]
    digest = _input_hash(pairs)
    if os.path.exists(outf) and not force:
        cached = _cached_mean(outf, pairs, digest)
        if cached is not None:
            return cached
        print(f"[rescore] {outf} does not match its input; judging again", flush=True)

    scores = score_rubric(pairs)
    got = [s for s in scores if s is not None]
    if not got:
        return None
    mean = st.mean(got)
    sd = st.stdev(got) if len(got) > 1 else 0.0
    with open(outf, "w") as fh:
        json.dump({"judge": f"{MODEL} strongreject-rubric", "n": len(got),
                   "n_total": len(scores), "mean_p_harmful": mean, "sd_p_harmful": sd,
                   "input_sha256": digest,
                   "results": [{"prompt": b, "deepseek_score": s}
                               for (b, _), s in zip(pairs, scores)]},
                  fh, indent=2)
    return mean


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+",
                    help="result files (step_N.json / base.json) or directories to search")
    ap.add_argument("--force", action="store_true", help="re-judge even if the output exists")
    args = ap.parse_args()

    files = [f for f in find_inputs(args.inputs)
             if not os.path.basename(f).startswith(PREFIX)]
    if not files:
        raise SystemExit("[rescore] no ClearHarm result files found")
    print(f"[rescore] judge={MODEL}, {len(files)} file(s)", flush=True)
    rows = []
    for inf in files:
        outf = output_path(inf)
        m = rescore(inf, outf, force=args.force)
        rows.append((inf, m))
        print(f"[rescore] {inf} -> {'--' if m is None else f'{m:.4f}'}", flush=True)

    print("\n| File | StrongREJECT (rubric) |")
    print("|:-|:-|")
    for inf, m in rows:
        print(f"| {inf} | {'--' if m is None else f'{m:.2f}'} |")


if __name__ == "__main__":
    main()
