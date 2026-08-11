#!/usr/bin/env bash
# 2xH100 DDP runs of the native full-image DINOv3+LoRA deep ensemble.
#
#   DINOv3-base (frozen) + LoRA on the last block + linear head
#   native resolution + per-batch width padding (DistributedWidthBucketSampler)
#   bf16 autocast, 5 experts x 30 epochs
#   resolution ramp 75% -> 50% -> 25% (2 epochs each), then full res
#   per-rank batch 96 (global 192), lr 2e-4, cosine (no warmup), patience 6
#
# Two legs (both DDP, 2xH100):
#   A  non-cluster: 5 experts on leave-one-out train subsets, no gate
#   B  cluster:     stage-1 embedders + 5 cluster experts (UMAP on val + KMeans),
#                   combined via plain/temp-scaled/gated selection on val
#
# Node prerequisites:
#   1. a conda env activated (torch cu12x, timm, peft, umap-learn, ...); the
#      script uses $CONDA_PREFIX/bin/python and .../bin/torchrun automatically
#   2. prepared challenge_data/ at $DATA (rsync from the dev box; runs/ not needed)
#   3. two GPUs visible:  python -c "import torch; print(torch.cuda.device_count())" == 2
#
# Run inside tmux:
#   tmux new-session -s edda "bash run_edda.sh"     # detach: Ctrl-b d
#   tmux attach -t edda                             # re-attach
#
# Output:
#   $OUT_NORMAL/experts/ (expert_seed0-4.pt), metrics_val.json, submission.csv
#   $OUT_CLUSTER/experts/, metrics_val.json, submission.csv
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Never let ~/.local pip packages (e.g. an unrelated torch) shadow the env's.
export PYTHONNOUSERSITE=1

# Resolve python/torchrun from the active conda env (or PY/TORCHRUN overrides).
# When a conda env is active, we REQUIRE torch from it -- no falling back to
# `command -v`, which can pick up a stale user-site torchrun.
if [ -z "${PY:-}" ]; then
    PY="${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"}"
    [ -z "$PY" ] && PY="$(command -v python)"
fi
if [ -z "${TORCHRUN:-}" ]; then
    TORCHRUN="${CONDA_PREFIX:+"$CONDA_PREFIX/bin/torchrun"}"
    [ -z "$TORCHRUN" ] && TORCHRUN="$(command -v torchrun)"
fi
[ -x "$PY" ] || {
    echo "ERROR: python not found ($PY). Activate your conda env, e.g."
    echo "  conda activate ss26"
    exit 1
}
[ -x "$TORCHRUN" ] || {
    echo "ERROR: torchrun not found in the active env ($CONDA_PREFIX)."
    echo "  Install torch into it, e.g."
    echo "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130"
    exit 1
}
echo ">>> using python:    $PY"
echo ">>> using torchrun:  $TORCHRUN"
RUNS="${RUNS:-$PROJECT_ROOT/runs}"
DATA="${DATA:-$PROJECT_ROOT/challenge_data}"
OUT="${OUT:-$RUNS/ensemble_dinov3_raw_native_edda_h100}"

[ -d "$DATA" ] || {
    echo "ERROR: data dir not found: $DATA"
    echo "  export DATA=/path/to/challenge_data"
    exit 1
}
# The prepared challenge_data layout (student/data.py):
#   challenge_data/{train,val}/images + labels.csv, test_public/images, class_mapping.json
for p in class_mapping.json train/images val/images test_public/images; do
    [ -e "$DATA/$p" ] || {
        echo "ERROR: expected '$DATA/$p' but it's missing."
        echo "  Data layout not prepared. Rsync the prepared challenge_data from the dev box:"
        echo "  rsync -av /home/alice/work/dtu_ss_26/challenge_data/ aliceschiavone@edda:/staff/aliceschiavone/ss26/challenge_data/"
        exit 1
    }
done

N_EXPERTS="${N_EXPERTS:-5}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-96}"     # per rank; global = 2 x 96 = 192
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-0}"
WORKERS="${WORKERS:-8}"
PATIENCE="${PATIENCE:-6}"
N_CLUSTERS="${N_CLUSTERS:-5}"
RAMP_SCALES="${RAMP_SCALES:-0.75 0.5 0.25}"
RAMP_EPOCHS="${RAMP_EPOCHS:-2}"

OUT_NORMAL="${OUT:-$RUNS/ensemble_dinov3_raw_native_edda_h100}"
OUT_CLUSTER="$RUNS/ensemble_dinov3_raw_native_edda_h100_cluster"

export PYTHONUNBUFFERED=1

run_leg() {
  local out="$1"; shift
  mkdir -p "$out"
  echo
  echo "=== launching (log: $out/train.log) ==="
  "$TORCHRUN" --nproc_per_node=2 --standalone \
    train_dinov3_ensemble.py \
    --data-root "$DATA" \
    --output-dir "$out" \
    --native --amp-bf16 \
    --lora-last-layers 1 \
    --n-experts "$N_EXPERTS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --warmup-epochs "$WARMUP" \
    --weight-decay 0.05 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --patience "$PATIENCE" \
    --num-workers "$WORKERS" \
    --ramp-scales $RAMP_SCALES \
    --ramp-epochs "$RAMP_EPOCHS" \
    "$@" \
    2>&1 | tee "$out/train.log"
}

echo "=== Leg A: non-cluster, $N_EXPERTS leave-one-out experts, no gate ==="
echo "    $EPOCHS epochs, ramp '$RAMP_SCALES' x$RAMP_EPOCHS epochs each, per-rank bs $BATCH (global $((BATCH * 2)))"
run_leg "$OUT_NORMAL" --no-gate

echo
echo "=== Leg B: cluster mode, UMAP on val + $N_CLUSTERS cluster experts + gate ==="
run_leg "$OUT_CLUSTER" --cluster-mode on --n-clusters "$N_CLUSTERS"

echo
echo ">>> Done. Leg A: $OUT_NORMAL/metrics_val.json  Leg B: $OUT_CLUSTER/metrics_val.json"
