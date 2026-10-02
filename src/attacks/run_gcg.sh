#!/usr/bin/env bash
# GCG suffix optimisation on ClearHarm with Claudini, for one model and one
# shard of samples. Output: <out_dir>/gcg/clearharm_ours/<model_tag>/sample_<id>_seed_0.json,
# consumed by src/attacks/gcg_regenerate.py --gcg-dir <out_dir>.
#
#   bash src/attacks/run_gcg.sh <model> <out_dir> [samples]
#
#   model    HF id (base model) or LoRA checkpoint dir; the patched model loader
#            merges the adapter into its base once and caches it under
#            $HF_HUB_CACHE/<rid>__step_<N>-merged (about 15 GB per 7-8B model).
#   out_dir  Claudini --results-dir for this model.
#   samples  comma-separated ClearHarm sample ids, or "all" (default) for the 40
#            ids in third_party/claudini/clearharm_ours.yaml.
#
# Env: CLAUDINI_DIR    claudini checkout at f97da47ca1f4cfcadb26ce54fd98ed4d960bd157
#                      with third_party/claudini/claudini_lora.patch applied (required)
#      CLAUDINI_PYTHON interpreter of claudini's environment (default: python)
#
# The paper ran the 40 samples as 5 shards of 8 (one GPU job each, all pointing
# at the same out_dir; finished samples are skipped on rerun):
#   for s in 1,3,6,7,8,9,10,11 13,16,17,18,19,20,21,22 23,24,25,26,27,29,30,31 \
#            32,34,35,36,37,38,39,40 41,42,43,45,46,48,49,51; do
#     bash src/attacks/run_gcg.sh "$NM_ROOT/lora-finetuned/<rid>/step_<N>" \
#         "$NM_ROOT/claudini_clearharm/<rid>_step<N>" "$s"
#   done
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,17p' "$0"; exit 1
fi
MODEL=$1
OUT=$(realpath -m "$2")
SAMPLE=${3:-all}
: "${CLAUDINI_DIR:?set CLAUDINI_DIR to the patched claudini checkout}"
PY=${CLAUDINI_PYTHON:-python}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$REPO/third_party/claudini/clearharm_ours.yaml"
# A local checkpoint dir must be absolute because we cd into CLAUDINI_DIR.
if [ -d "$MODEL" ]; then MODEL=$(realpath "$MODEL"); fi

sample_flag=(--sample "$SAMPLE")
if [ "$SAMPLE" = "all" ]; then sample_flag=(); fi

cd "$CLAUDINI_DIR"
exec "$PY" -m claudini.run_bench "$CONFIG" \
    --method gcg --model "$MODEL" "${sample_flag[@]}" --seed 0 \
    --max-flops 1e17 --results-dir "$OUT"
