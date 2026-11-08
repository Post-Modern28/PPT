#!/bin/bash
#SBATCH --job-name=evaluate_all
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/evaluate_all_upscaled.out
#SBATCH --error=logs/evaluate_all_upscaled.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt
nvidia-smi
cd ~/benchmarking/PPT

echo "Evaluating all epochs..."

for OUT_DIR in results/test/test_*; do
    epoch=$(basename "$OUT_DIR" | cut -d'_' -f2)
    IMG_PATH=${OUT_DIR}/images

    if [ -d "$IMG_PATH" ]; then
        echo "=== Evaluating epoch $epoch ==="
        python evaluation_edited.py --image_path "$IMG_PATH"
    else
        echo "Skipping epoch $epoch (no images found)"
    fi
done

echo "Done evaluating all epochs."

