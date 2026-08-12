#!/usr/bin/env bash
# Single-expert DINOv2-Giant FULL-LoRA probe (all 40 blocks).
#
#   frozen Giant + LoRA on ALL transformer blocks (--lora-last-layers 0)
#     + linear head; 1 expert trained on ALL train data (no LOO hold-out)
#   fixed 518x518 full-image, bf16 autocast, 2xH100 DDP
#   per-rank batch 48 x 2 grad-accum (effective global 192), lr 1e-4,
#   cosine + 1-epoch warmup, 10 epochs, label smoothing 0.05 + mixup 0.1,
#   early stop on val NLL (patience 2)
#
# Tests the hypothesis that the frozen-last-block regime (lora-last-layers 1,
# ~290k trainable params) is the bottleneck, and is one member of the
# 3-architecture ensemble (Giant + ConvNeXtV2-Huge + RegNetY-1280), merged
# post-hoc by cluster_entropy_temp.py --run-dirs. Output goes to a fresh dir so
# the existing leg A/B runs are untouched:
#   $RUNS/ensemble_dinov2_giant_518_lora_all_single_10e/
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
OUT="${OUT:-$RUNS/ensemble_dinov2_giant_518_lora_all_single_10e}"
BATCH="${BATCH:-48}"        # per rank; global effective = 48 x 2 ranks x 2 accum = 192
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"

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
echo ">>> single-expert FULL-LoRA (all 40 blocks) run: 1 expert x 10 epochs"
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
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --warmup-epochs 1 \
    --weight-decay 0.05 \
    --label-smoothing 0.05 \
    --mixup-alpha 0.1 \
    --patience 2 \
    --num-workers "${WORKERS:-8}" \
    --no-gate \
    2>&1 | tee "$OUT/train.log"

echo
echo ">>> Done. metrics: $OUT/metrics_val.json"
