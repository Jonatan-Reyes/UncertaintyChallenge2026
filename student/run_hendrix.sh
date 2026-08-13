#!/bin/bash
#SBATCH --job-name=lora-ensemble-dinov3
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=32000M
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate uncertainty

hostname
echo "cuda Devices"
echo $CUDA_VISIBLE_DEVICES

# SLURM copies this script into a spool dir before running it, so
# ${BASH_SOURCE[0]} doesn't point at the repo — use SLURM_SUBMIT_DIR instead.
# Assumes you submit with `sbatch student/run_hendrix.sh` from the repo root.
#cd "$SLURM_SUBMIT_DIR"

/home/wxp878/miniconda3/envs/uncertainty/bin/python -m student.train \
    --data-root challenge_data \
    --output-dir results \
    --pretrained \
    --backbones vit_large_patch14_dinov2.lvd142m eva02_large_patch14_224.mim_in22k \
    --epochs 30 --batch-size 64 \
    --lr 1e-4 --head-lr 1e-3 \
    --num-unfrozen-layers 0 \
    --alpha 1.0 \
    --patience 3
