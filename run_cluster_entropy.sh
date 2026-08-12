#!/usr/bin/env bash
# Post-hoc cluster class-entropy temperature on a finished training run (edda).
#
# No retraining. Runs cluster_entropy_temp.py against a run dir that already has
# experts/expert_seed*.pt + config.json (e.g. the finished cluster leg):
#
#   * embeds train+val with the trained experts (default: the single best
#     expert on val NLL; --embed-source average uses the mean of all)
#   * UMAP(2d) + KMeans(10) on the pooled train+val embeddings
#   * per-cluster class entropy -> per-cluster temperature (mixed clusters = more
#     uncertain), fitted on val and guarded by within-val CV
#   * writes <run-dir>/posthoc/submission.csv (temperature applied to the
#     averaged logits) + report.json
#
# Everything expensive is cached under <run-dir>/posthoc/, so re-running with
# different temperature parameters (ALPHA_GRID, ENTROPY, TEMP_CLIP_*) is fast.
#
# Usage (in tmux on a node with a free GPU):
#   tmux new-session -s ent "bash run_cluster_entropy.sh"
#
# Env overrides:
#   RUN_DIR         run dir with config.json + experts/      (default: <repo>/../runs/ensemble_dinov2_giant_518_edda_h100_cluster)
#   DATA            override the data_root from config.json
#   EMBED_SOURCE    best|average                             (default: best)
#   K               number of clusters                       (default: 10)
#   ALPHA_GRID      candidate alpha values                   (default: "0 0.25 0.5 0.75 1 1.5")
#   ENTROPY         shannon|gini                             (default: shannon)
#   TEMP_CLIP_MIN   lower T clip factor                      (default: 0.3)
#   TEMP_CLIP_MAX   upper T clip factor                      (default: 3.0)
#   METHOD          auto|cluster_entropy_T|temp_scaled_average|plain_average (default: auto)
#   PY              python binary override
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ -z "${PY:-}" ]; then
    PY="${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"}"
    [ -z "$PY" ] && PY="$(command -v python)"
fi
[ -x "$PY" ] || {
    echo "ERROR: python not found ($PY). Activate your conda env, e.g. conda activate ss226"
    exit 1
}
echo ">>> using python: $PY"

RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/ensemble_dinov2_giant_518_edda_h100_cluster}"
EMBED_SOURCE="${EMBED_SOURCE:-best}"
K="${K:-10}"
ALPHA_GRID="${ALPHA_GRID:-0 0.25 0.5 0.75 1 1.5}"
ENTROPY="${ENTROPY:-shannon}"
TEMP_CLIP_MIN="${TEMP_CLIP_MIN:-0.3}"
TEMP_CLIP_MAX="${TEMP_CLIP_MAX:-3.0}"
METHOD="${METHOD:-auto}"

[ -f "$RUN_DIR/config.json" ] || {
    echo "ERROR: no config.json in $RUN_DIR"
    echo "  export RUN_DIR=/path/to/run_dir"
    exit 1
}
[ -d "$RUN_DIR/experts" ] && ls "$RUN_DIR"/experts/expert_seed*.pt >/dev/null 2>&1 || {
    echo "ERROR: no expert_seed*.pt in $RUN_DIR/experts"
    exit 1
}

DATA_ARGS=()
if [ -n "${DATA:-}" ]; then
    for p in class_mapping.json train/images val/images test_public/images; do
        [ -e "$DATA/$p" ] || {
            echo "ERROR: expected '$DATA/$p' but it's missing."
            exit 1
        }
    done
    DATA_ARGS=(--data-root "$DATA")
fi

echo ">>> run-dir:       $RUN_DIR"
echo ">>> embed-source:  $EMBED_SOURCE"
echo ">>> K:             $K"
echo ">>> alpha grid:    $ALPHA_GRID"

"$PY" cluster_entropy_temp.py \
    --run-dir "$RUN_DIR" \
    --embed-source "$EMBED_SOURCE" \
    --k "$K" \
    --alpha-grid $ALPHA_GRID \
    --entropy "$ENTROPY" \
    --temp-clip-min "$TEMP_CLIP_MIN" \
    --temp-clip-max "$TEMP_CLIP_MAX" \
    --method "$METHOD" \
    "${DATA_ARGS[@]}"

echo
echo ">>> report:     $RUN_DIR/posthoc/report.json"
echo ">>> submission: $RUN_DIR/posthoc/submission.csv"
