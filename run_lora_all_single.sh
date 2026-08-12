#!/usr/bin/env bash
# Single-expert DINOv2-Giant FULL-LoRA probe (all 40 blocks).
#
#   frozen Giant + LoRA on ALL transformer blocks (--lora-last-layers 0)
#     + linear head; 1 expert trained on ALL train data (no LOO hold-out)
#   fixed 518x518 full-image, bf16 autocast, 2xH100 DDP
#   per-rank batch 96 (global 192), lr 1e-4, cosine + 3-epoch warmup,
#   30 epochs, label smoothing 0.1 + mixup 0.2, early stop on val NLL (patience 8)
#
# Tests the hypothesis that the frozen-last-block regime (lora-last-layers 1,
# ~290k trainable params) is the bottleneck. Output goes to a fresh dir so the
# existing leg A/B runs are untouched:
#   $RUNS/ensemble_dinov2_giant_518_lora_all_single_30e/
#
# Run inside tmux:
#   tmux new-session -s loraall "bash run_lora_all_single.sh"
#   # detach: Ctrl-b d     re-attach: tmux attach -t loraall
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Never let ~/.local pip packages (e.g. an unrelated torch) shadow the env's.
export PYTHONNOUSERSITE=1

if [ -z "${PY:-}" ]; then
    PY="${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"}"
    [ -z "$PY" ] && PY="$(command -v python)"
fi
if [ -z "${TORCHRUN:-}" ]; then
    TORCHRUN="${CONDA_PREFIX:+"$CONDA_PREFIX/bin/torchrun"}"
    [ -z "$TORCHRUN" ] && TORCHRUN="$(command -v torchrun)"
fi
[ -x "$PY" ] || {
    echo "ERROR: python not found ($PY). Activate your conda env, e.g. conda activate ss26"
    exit 1
}
[ -x "$TORCHRUN" ] || {
    echo "ERROR: torchrun not found in the active env ($CONDA_PREFIX)."
    echo "  Install torch into it (see run_edda.sh header)."
    exit 1
}
echo ">>> using python:    $PY"
echo ">>> using torchrun:  $TORCHRUN"

RUNS="${RUNS:-$PROJECT_ROOT/runs}"
DATA="${DATA:-$PROJECT_ROOT/challenge_data}"
OUT="${OUT:-$RUNS/ensemble_dinov2_giant_518_lora_all_single_30e}"

[ -d "$DATA" ] || {
    echo "ERROR: data dir not found: $DATA"
    echo "  export DATA=/path/to/challenge_data"
    exit 1
}
for p in class_mapping.json train/images val/images test_public/images; do
    [ -e "$DATA/$p" ] || {
        echo "ERROR: expected '$DATA/$p' but it's missing (prepared challenge_data layout)."
        exit 1
    }
done

export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

echo
echo ">>> single-expert FULL-LoRA (all 40 blocks) run: 1 expert x 30 epochs"
echo ">>> log: $OUT/train.log  (tail -f to watch)"
"$TORCHRUN" --nproc_per_node=2 --standalone \
    train_dinov3_ensemble.py \
    --data-root "$DATA" \
    --output-dir "$OUT" \
    --backbone "${BACKBONE:-vit_giant_patch14_dinov2.lvd142m}" \
    --img-size 518 --full-image \
    --amp-bf16 \
    --lora-last-layers 0 \
    --lora-r 8 --lora-alpha 16 \
    --n-experts 1 \
    --epochs 30 \
    --batch-size 96 \
    --lr 1e-4 \
    --warmup-epochs 3 \
    --weight-decay 0.05 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --patience 8 \
    --num-workers "${WORKERS:-8}" \
    --no-gate \
    2>&1 | tee "$OUT/train.log"

echo
echo ">>> Done. metrics: $OUT/metrics_val.json"
