#!/usr/bin/env bash
# Train both ablation runs on FULL images (uncropped, native-ish resolution).
#
#   A: 10 full-data deep-ensemble experts  (--cluster-mode off, no gating)
#   B: 10 cluster-specialist experts       (--cluster-mode on, hold-out sets from
#      corner clustering + distance soft-gate)
#
# Both use the exact same recipe as the validated QC run:
#   DINOv3-base (frozen) + LoRA on the last 2 blocks + linear head
#   full image at IMG_SIZE (Resize, NO random-resized crop)
#   CE label smoothing 0.1 + mixup 0.2, AdamW lr 1e-4 / wd 0.05, cosine+warmup,
#   early stop on val NLL (patience 4), 15 epochs, batch 32.
#
# IMG_SIZE = 448 because every raw image has min side >= 448, so the whole
# scene is kept at native resolution (larger images are downscaled).
#
# Usage:  bash run_full_training.sh
# Output: $RUNS/ensemble_dinov3_raw_full_448/       (A: metrics_val.json, submission.csv)
#         $RUNS/ensemble_dinov3_raw_clusters_448/   (B: metrics_val.json, submission.csv)
set -euo pipefail
cd "$(dirname "$0")"

PY="$HOME/miniconda3/envs/ss/bin/python"
RUNS="/home/alice/work/dtu_ss_26/runs"
DATA="/home/alice/work/dtu_ss_26/challenge_data"
CORNERS="/home/alice/work/dtu_ss_26/challenge_data_prep_corners64"

IMG_SIZE=448
EPOCHS=20
BATCH=32
WORKERS=8
LORA_LAST=3

OUT_A="$RUNS/ensemble_dinov3_raw_full_448"
OUT_B="$RUNS/ensemble_dinov3_raw_clusters_448"
mkdir -p "$OUT_A" "$OUT_B"

echo ">>> [A] 10 full-data experts, full image ${IMG_SIZE}px, LoRA last ${LORA_LAST} blocks"
"$PY" -u train_dinov3_ensemble.py \
  --data-root "$DATA" \
  --corner-root "$CORNERS" \
  --output-dir "$OUT_A" \
  --img-size "$IMG_SIZE" --full-image \
  --lora-last-layers "$LORA_LAST" \
  --n-experts 5 --epochs "$EPOCHS" --batch-size "$BATCH" \
  --num-workers "$WORKERS" --no-gate \
  2>&1 | tee "$OUT_A/train.log"

echo
echo ">>> [B] 10 cluster-specialist experts, full image ${IMG_SIZE}px, LoRA last ${LORA_LAST} blocks"
"$PY" -u train_dinov3_ensemble.py \
  --data-root "$DATA" \
  --corner-root "$CORNERS" \
  --output-dir "$OUT_B" \
  --img-size "$IMG_SIZE" --full-image \
  --lora-last-layers "$LORA_LAST" \
  --cluster-mode on --n-clusters 5 --epochs "$EPOCHS" --batch-size "$BATCH" \
  --num-workers "$WORKERS" \
  2>&1 | tee "$OUT_B/train.log"

echo
echo ">>> Both runs finished."
echo "    A: $OUT_A/metrics_val.json"
echo "    B: $OUT_B/metrics_val.json"
