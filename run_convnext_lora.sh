#!/usr/bin/env bash
# Single-expert ConvNeXt V2 Huge LoRA probe (all 1x1 convs), 518x518.
#
#   frozen ConvNeXtV2-Huge + LoRA on every kernel-size-1 Conv2d (the pointwise
#     channel-mixing layers) + linear head; 1 expert trained on ALL train data
#   fixed 518x518 full-image, bf16 autocast, 2xH100 DDP
#   per-rank batch 96 (global 192), lr 1e-4, cosine + 1-epoch warmup,
#   10 epochs, label smoothing 0.05 + mixup 0.1, early stop on val NLL (patience 2)
#
# One member of the 3-architecture ensemble (Giant + ConvNeXtV2-Huge +
# RegNetY-1280), merged post-hoc by cluster_entropy_temp.py --run-dirs.
# Output goes to a fresh dir:
#   $RUNS/ensemble_convnextv2_huge_518_lora_10e/
#
# Run inside tmux:
#   tmux new-session -s convnext "bash run_convnext_lora.sh"
#   # detach: Ctrl-b d     re-attach: tmux attach -t convnext
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
OUT="${OUT:-$RUNS/ensemble_convnextv2_huge_518_lora_10e}"

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
echo ">>> ConvNeXtV2-Huge LoRA (1x1 convs) run: 1 expert x 10 epochs @518px"
echo ">>> log: $OUT/train.log  (tail -f to watch)"
"$TORCHRUN" --nproc_per_node=2 --standalone \
    train_dinov3_ensemble.py \
    --data-root "$DATA" \
    --output-dir "$OUT" \
    --backbone "${BACKBONE:-convnextv2_huge.fcmae_ft_in22k_in1k_512}" \
    --img-size 518 --full-image \
    --amp-bf16 \
    --lora-last-layers 0 \
    --lora-r 8 --lora-alpha 16 \
    --n-experts 1 \
    --epochs 10 \
    --batch-size 96 \
    --lr 1e-4 \
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
