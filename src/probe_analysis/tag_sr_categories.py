"""Add an 8-category `category` field to every result in the figure1 runs'
StrongREJECT step files, filled by the gpt-5-mini judge.

For each figure1 run, for each live_downstream/<run>/strongreject/step_*.json,
every results[] entry (prompt / completion / score) gets a new "category" key
(one of llm_judge_classify.CATEGORIES) from call_judge(). Written IN PLACE.

Resumable: results that already carry a "category" are skipped, and each file is
saved atomically as soon as its 100 completions are judged, so an interrupted run
picks up where it left off. Reuses the taxonomy / prompt / model from
llm_judge_classify verbatim.

    NM_ROOT=/data/new_master \
        python3 src/probe_analysis/tag_sr_categories.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_judge_classify import (  # noqa: E402
    DEFAULT_MODEL, LIVE_ROOT, MAX_WORKERS, call_judge, get_client,
)

# The 12 figure1 runs (Mistral + abliterated Llama, frozen/contupd/reinit x probe/poly).
FIG1_RUN_IDS = [
    "RUN_023", "RUN_016", "RUN_015", "RUN_005", "RUN_018", "RUN_003",
    "RUN_021", "RUN_020", "RUN_008", "RUN_002", "RUN_019", "RUN_014",
]
MODEL = os.environ.get("JUDGE_MODEL", DEFAULT_MODEL)


def step_files(run):
    d = LIVE_ROOT / run / "strongreject"
    return sorted(d.glob("step_*.json"),
                  key=lambda p: int(p.stem.split("_")[1]))


def process_file(client, path):
    """Judge every result missing a category; save the file if anything changed.
    Returns the number of completions newly judged."""
    d = json.load(open(path))
    results = d.get("results", [])
    todo = [i for i, r in enumerate(results) if not r.get("category")]
    if not todo:
        return 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for i in todo:
            r = results[i]
            try:
                sc = float(r.get("score", 0.0))
            except (TypeError, ValueError):
                sc = 0.0
            futs[ex.submit(call_judge, client, MODEL, r.get("prompt", ""),
                           r.get("completion", ""), sc)] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i]["category"] = fut.result().get("category")
            except Exception as e:  # noqa: BLE001
                results[i]["category"] = None
                print(f"      [warn] {path.name} idx {i}: {e}", flush=True)
    tmp = path.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, path)
    return len(todo)


def main():
    client = get_client()
    total = 0
    for run in FIG1_RUN_IDS:
        files = step_files(run)
        judged_run = 0
        for p in files:
            n = process_file(client, p)
            judged_run += n
            if n:
                print(f"  {run} {p.name}: judged {n}", flush=True)
        print(f"[run done] {run}: {judged_run} completions judged "
              f"across {len(files)} step files", flush=True)
        total += judged_run
    print(f"[all done] {total} completions newly judged; "
          f"model={MODEL}", flush=True)


if __name__ == "__main__":
    main()
