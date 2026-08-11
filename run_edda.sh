#!/usr/bin/env bash
# 2xH100 DDP run of the native full-image DINOv3+LoRA deep ensemble.
#
#   DINOv3-base (frozen) + LoRA on the last block + linear head
#   native resolution + per-batch width padding (DistributedWidthBucketSampler)
#   bf16 autocast, 5 experts x 20 epochs
#   per-rank batch 96 (global 192), lr 2e-4, warmup 3 epochs, cosine, patience 6
#
# Node prerequisites:
#   1. a conda env activated (torch cu12x, timm, peft, umap-learn, ...); the
#      script uses $CONDA_PREFIX/bin/python and .../bin/torchrun automatically
#   2. raw challenge_data/ present at $DATA (rsync from the dev box; runs/ not needed)
#   3. two GPUs visible:  python -c "import torch; print(torch.cuda.device_count())" == 2
#
# Run inside tmux:
#   tmux new-session -s edda "bash run_edda.sh"     # detach: Ctrl-b d
#   tmux attach -t edda                             # re-attach
#
# Output: $OUT/experts/ (expert_seed0-4.pt), $OUT/metrics_val.json, $OUT/submission.csv
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="/users/aliceschiavone/ss26/"

# Resolve python/torchrun from the active conda env (or PY/TORCHRUN overrides).
if [ -z "${PY:-}" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
fi
if [ -z "${PY:-}" ]; then
    PY="$(command -v python)"
fi
if [ -z "${TORCHRUN:-}" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/torchrun" ]; then
    TORCHRUN="$CONDA_PREFIX/bin/torchrun"
fi
if [ -z "${TORCHRUN:-}" ]; then
    TORCHRUN="$(command -v torchrun)"
fi
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
    echo "ERROR: could not find a python in the active conda env. Activate it, e.g."
    echo "  conda activate ss26"
    exit 1
fi
RUNS="${RUNS:-$PROJECT_ROOT/runs}"
DATA="${DATA:-$PROJECT_ROOT/challenge_data}"
OUT="${OUT:-$RUNS/ensemble_dinov3_raw_native_edda_h100}"

[ -d "$DATA" ] || {
    echo "ERROR: data dir not found: $DATA"
    echo "  export DATA=/path/to/challenge_data"
    exit 1
}

N_EXPERTS="${N_EXPERTS:-5}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-96}"     # per rank; global = 2 x 96 = 192
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-3}"
WORKERS="${WORKERS:-8}"
PATIENCE="${PATIENCE:-6}"

mkdir -p "$OUT"

echo ">>> 2xH100 DDP native run: $N_EXPERTS experts x $EPOCHS epochs, per-rank bs $BATCH (global $((BATCH * 2)))"
"$TORCHRUN" --nproc_per_node=2 --standalone \
  "$PY" -u train_dinov3_ensemble.py \
  --data-root "$DATA" \
  --output-dir "$OUT" \
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
  --no-gate \
  2>&1 | tee "$OUT/train.log"

echo
echo ">>> Done: $OUT/metrics_val.json  $OUT/submission.csv"
