#!/bin/bash
#SBATCH --job-name=lora-ensemble-dinov3
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=32000M
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate uncertainty

hostname
echo $CUDA_VISIBLE_DEVICES

# SLURM copies this script into a spool dir before running it, so
# ${BASH_SOURCE[0]} doesn't point at the repo — use SLURM_SUBMIT_DIR instead.
# Assumes you submit with `sbatch student/run_hendrix.sh` from the repo root.
cd "$SLURM_SUBMIT_DIR"

python -m student.train \
    --data-root challenge_data \
    --output-dir results \
    --pretrained \
    --backbone vit_small_patch16_dinov3.lvd1689m \
    --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
    --num-lora-members 4 \
    --epochs 30 --batch-size 32 \
    --lr 1e-4 --head-lr 1e-3 \
    --patience 3
