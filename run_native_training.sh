#!/usr/bin/env bash
# Native-resolution full-image run (no resize, no crop).
#
#   DINOv3-base (frozen) + LoRA on the last 3 blocks + linear head
#   images keep their native size (448 tall, 560-796 wide); each batch is padded
#   to the batch max via dynamic_pad_collate + a width-bucketed sampler
#   intensity augmentation biased to contrast/brightness (data is ~95% grayscale)
#   5 experts, 20 epochs, batch 48, AdamW lr 1e-4 / wd 0.05, cosine+warmup(2),
#   label smoothing 0.1 + mixup 0.2, early stop on val NLL (patience 6).
#
# Run inside tmux so it survives shell disconnects:
#   tmux new-session -s native "bash run_native_training.sh"
#   # detach:  Ctrl-b d     re-attach:  tmux attach -t native
#
# Output: $RUNS/ensemble_dinov3_raw_native_l3_20e/  (experts/, metrics_val.json, submission.csv)
set -euo pipefail
cd "$(dirname "$0")"

PY="$HOME/miniconda3/envs/ss/bin/python"
RUNS="${RUNS:-/home/alice/work/dtu_ss_26/runs}"
DATA="/home/alice/work/dtu_ss_26/challenge_data"
OUT="${OUT:-$RUNS/ensemble_dinov3_raw_native_l3_20e}"

N_EXPERTS="${N_EXPERTS:-5}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-48}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"

mkdir -p "$OUT"

echo ">>> native full-image run: $N_EXPERTS experts x $EPOCHS epochs, batch $BATCH"
echo ">>> LoRA last 3 blocks, native res + per-batch padding, no gate"
echo ">>> log: $OUT/train.log  (tail -f to watch)"
"$PY" -u train_dinov3_ensemble.py \
  --data-root "$DATA" \
  --output-dir "$OUT" \
  --native \
  --lora-last-layers 3 \
  --n-experts "$N_EXPERTS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2 \
  --patience 6 \
  --num-workers "$WORKERS" \
  --no-gate \
  2>&1 | tee "$OUT/train.log"

echo
echo ">>> Done: $OUT/metrics_val.json  $OUT/submission.csv"
