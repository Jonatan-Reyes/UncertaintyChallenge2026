#!/usr/bin/env bash
# Build an ensemble submission from already-trained run dirs (no training).
#
# Each run dir becomes ONE member: its experts are plain-averaged into a single
# member logits vector, members are stacked, then post-hoc temperature scaling /
# cluster class-entropy temperature is fit on val and applied to test splits
# (cluster_entropy_temp.py --run-dirs).
#
# Default members: the 2 convnet LoRA probes + one already-trained Giant leg, so
# you can see the ensemble's val performance while the new Giant full-LoRA probe
# is still training. Override any path:
#   RUN_CONVNEXT=<dir> RUN_REGYNET=<dir> RUN_GIANT=<dir> bash run_combine.sh
# To combine only the two convnets: add COMBINE_DIRS_EXTRA handling below, or
#   python -u cluster_entropy_temp.py --run-dirs <convnext> <regnety> ...
#
# Output: report.json / chosen.json / submission.csv under the posthoc/ dir of
# the best member (by val NLL). Everything expensive is cached there.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONNOUSERSITE=1

PY="${PY:-$CONDA_PREFIX/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
[ -x "$PY" ] || {
    echo "ERROR: python not found. Activate your conda env, e.g. conda activate ss26"
    exit 1
}
echo ">>> using python:    $PY"

RUNS="${RUNS:-$PROJECT_ROOT/runs}"
DATA="${DATA:-$PROJECT_ROOT/challenge_data}"
RUN_CONVNEXT="${RUN_CONVNEXT:-$RUNS/ensemble_convnextv2_huge_518_lora_10e}"
RUN_REGYNET="${RUN_REGYNET:-$RUNS/ensemble_regnety1280_518_lora_10e}"
RUN_GIANT="${RUN_GIANT:-$RUNS/ensemble_dinov2_giant_518_edda_h100}"

for d in "$RUN_CONVNEXT" "$RUN_REGYNET" "$RUN_GIANT"; do
    [ -f "$d/config.json" ] || {
        echo "ERROR: run dir missing config.json: $d"
        echo "  Set RUN_CONVNEXT / RUN_REGYNET / RUN_GIANT to trained run dirs."
        exit 1
    }
done

echo ">>> combining members:"
echo "    $RUN_CONVNEXT"
echo "    $RUN_REGYNET"
echo "    $RUN_GIANT"
"$PY" -u cluster_entropy_temp.py \
    --run-dirs "$RUN_CONVNEXT" "$RUN_REGYNET" "$RUN_GIANT" \
    --data-root "$DATA" \
    --k 10 \
    --num-workers "${WORKERS:-8}" \
    "$@"
