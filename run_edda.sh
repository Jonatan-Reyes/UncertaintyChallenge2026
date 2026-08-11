#!/usr/bin/env bash
# 2xH100 DDP run of the native full-image DINOv3+LoRA deep ensemble.
#
#   DINOv3-base (frozen) + LoRA on the last block + linear head
#   native resolution + per-batch width padding (DistributedWidthBucketSampler)
#   bf16 autocast, 5 experts x 20 epochs
#   per-rank batch 96 (global 192), lr 2e-4, warmup 3 epochs, cosine, patience 6
#
# Node prerequisites:
#   1. conda env `ss` recreated from ss_env.yml (torch cu12x, timm, peft, ...)
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

PY="$HOME/miniconda3/envs/ss/bin/python"
TORCHRUN="${TORCHRUN:-$HOME/miniconda3/envs/ss/bin/torchrun}"
RUNS="${RUNS:-/home/alice/work/dtu_ss_26/runs}"
DATA="${DATA:-/home/alice/work/dtu_ss_26/challenge_data}"
OUT="${OUT:-$RUNS/ensemble_dinov3_raw_native_edda_h100}"

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
