#!/bin/bash
#SBATCH --job-name=test_missing
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/test_missing.out
#SBATCH --error=logs/test_missing.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt
nvidia-smi
cd ~/benchmarking/PPT

echo "Running test only on missing epochs..."

for ckpt in checkpoints/test/*_net_G.pth; do
    epoch=$(basename "$ckpt" | cut -d'_' -f1)
    OUT_DIR=results/test/test_${epoch}

    if [ ! -d "$OUT_DIR" ]; then
        echo "=== Running test for epoch $epoch ==="
        python test.py \
            --dataroot ~/benchmarking/datasets/HIT/HIT/PAX5/ \
            --name test \
            --phase test \
            --how_many 163 \
            --serial_batches \
            --which_epoch $epoch
    else
        echo "Skipping epoch $epoch (already tested)"
    fi
done

echo "Done testing missing epochs."

