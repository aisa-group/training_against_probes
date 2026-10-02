"""ClearHarm GCG eval: regenerate from Claudini suffixes and score.

Claudini (src/attacks/run_gcg.sh) optimises a 30-token suffix per ClearHarm
query and writes <gcg_dir>/gcg/clearharm_ours/<model_tag>/sample_<id>_seed_0.json,
but only reports target-match. For every finished sample this script rebuilds
the attacked input Claudini used (f"{query} {best_string}" in one user turn),
applies the chat template, generates 512 tokens greedily with the same base
model (+ LoRA), and scores the response against the ORIGINAL query with the
fine-tuned StrongREJECT judge, so the column is comparable to direct/prefill.

Only finished samples are scored; rerunning picks up newly finished ones.
Output layout matches src/attacks/clearharm_direct_prefill.py with mode "gcg":
  <out>/<rid>/clearharm_gcg/step_<N>.json   or   <out>/<name>/clearharm_gcg/base.json

    python src/attacks/gcg_regenerate.py --rid <rid> --step <N> --gcg-dir <claudini out>
    python src/attacks/gcg_regenerate.py --ckpt <a> --gcg-dir <a_out> --ckpt <b> --gcg-dir <b_out>
    python src/attacks/gcg_regenerate.py --base-model <hf_id> --name <name> --gcg-dir <out>
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SRC, "probes"))
sys.path.insert(0, SRC)

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from asr.judges import score_strongreject_batch  # noqa: E402
from attacks.clearharm_direct_prefill import (  # noqa: E402
    GPU_MEMORY_UTILIZATION, LORA_RANK, MAX_NEW_TOKENS, build_targets)
from utils.finetune_utils import apply_chat_template  # noqa: E402
from utils.vllm_utils import generate_vllm_subprocess  # noqa: E402

METHOD = "gcg"
TRACK = "clearharm_ours"  # Claudini track = config file stem
SEED = 0


def clearharm_examples():
    ds = load_dataset("AlignmentResearch/ClearHarm", split="train")
    return [row["content"][0] for row in ds]


def load_suffixes(gcg_dir):
    """{sample_id: best_string} for every finished sample. Unreadable or
    half-written files (shards may still be running) are skipped."""
    out = {}
    pattern = os.path.join(gcg_dir, METHOD, TRACK, "*", f"sample_*_seed_{SEED}.json")
    for f in glob.glob(pattern):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        sid = d.get("sample_id")
        if sid is not None and sid not in out and d.get("best_string"):
            out[sid] = d["best_string"]
    return out


def run(target, gcg_dir, examples, force=False):
    base, tok_name, lora, outroot, stem = target
    outdir = os.path.join(outroot, "clearharm_gcg")
    out = os.path.join(outdir, f"{stem}.json")
    sfx = load_suffixes(gcg_dir)
    if not sfx:
        print(f"[gcg] {gcg_dir}: no finished suffixes yet, skip", flush=True)
        return
    if os.path.exists(out) and not force:
        with open(out) as f:
            done = {r["sample_id"] for r in json.load(f)["results"]}
        if done >= set(sfx):
            print(f"[gcg] {out}: {len(done)} samples up to date, skip", flush=True)
            return
    os.makedirs(outdir, exist_ok=True)
    sids = sorted(sfx)
    # Claudini's ClearHarmSource maps sample_id -> examples[sample_id % len].
    queries = [examples[s % len(examples)] for s in sids]
    attacked = [f"{q} {sfx[s]}" for q, s in zip(queries, sids)]
    tok = AutoTokenizer.from_pretrained(tok_name)
    formatted = apply_chat_template(attacked, tok)
    print(f"[gcg] {gcg_dir} base={base} lora={lora} n={len(sids)}", flush=True)
    completions = generate_vllm_subprocess(
        formatted_prompts=formatted, model_name=base, tokenizer_name=tok_name,
        lora_path=lora, max_new_tokens=MAX_NEW_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, lora_rank=LORA_RANK)
    scores = [float(x) for x in score_strongreject_batch(queries, completions)]
    mean = st.mean(scores)
    sd = st.stdev(scores) if len(scores) > 1 else 0.0
    with open(out, "w") as f:
        json.dump({"model": base, "lora": lora, "mode": "gcg", "dataset": "ClearHarm",
                   "n_prompts": len(sids), "mean_p_harmful": mean, "sd_p_harmful": sd,
                   "results": [{"sample_id": s, "query": q, "attacked": a,
                                "response": c, "score": sc}
                               for s, q, a, c, sc in zip(sids, queries, attacked,
                                                         completions, scores)]},
                  f, indent=2)
    print(f"[gcg] {stem}: SR = {mean:.4f} +- {sd:.4f}  (n={len(sids)}) -> {out}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gcg-dir", action="append", default=[], required=True,
                    help="Claudini --results-dir of a model (repeatable, one per model, "
                         "in the order of --ckpt, then --rid, or the base model)")
    ap.add_argument("--base-model", help="HF id of the base model; evaluated on its own if "
                    "no checkpoint is given, otherwise overrides the adapter's base")
    ap.add_argument("--ckpt", action="append", default=[],
                    help="LoRA checkpoint dir (repeatable)")
    ap.add_argument("--rid", help="run id under $NM_ROOT/lora-finetuned (with --step)")
    ap.add_argument("--step", type=int, help="checkpoint step for --rid")
    ap.add_argument("--tokenizer", help="tokenizer id (default: base model, or "
                    "Llama-3-8B-Instruct for heretic bases)")
    ap.add_argument("--name", help="output dir name for a base model (default: last "
                    "component of --base-model)")
    ap.add_argument("--out", help="output root (default: $NM_ROOT/live_downstream for "
                    "checkpoints, $NM_ROOT/clearharm_sr for a base model)")
    ap.add_argument("--force", action="store_true", help="regenerate even if up to date")
    args = ap.parse_args(argv)
    if (args.rid is None) != (args.step is None):
        ap.error("--rid and --step must be given together")
    if not (args.ckpt or args.rid or args.base_model):
        ap.error("give --base-model, --ckpt or --rid/--step")
    return args


def main():
    args = parse_args()
    targets = build_targets(args)
    if len(args.gcg_dir) != len(targets):
        raise SystemExit(f"[gcg] got {len(args.gcg_dir)} --gcg-dir for {len(targets)} model(s)")
    examples = clearharm_examples()
    print(f"[gcg] {len(examples)} ClearHarm examples, {len(targets)} model(s)", flush=True)
    for t, gdir in zip(targets, args.gcg_dir):
        run(t, gdir, examples, force=args.force)
    print("[gcg] done", flush=True)


if __name__ == "__main__":
    main()
