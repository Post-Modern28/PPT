#!/bin/bash
#SBATCH --job-name=test_ppt
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/test_ppt_cascade.out
#SBATCH --error=logs/test_ppt_cascade.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt
nvidia-smi
cd ~/benchmarking/PPT

echo "Testing all epochs..."

for ckpt in checkpoints/test/*_net_G.pth; do
    epoch=$(basename $ckpt | cut -d'_' -f1)
    echo "=== Testing epoch $epoch ==="

    python test.py \
        --dataroot ~/benchmarking/datasets/HIT/HIT/PAX5/ \
        --name test \
        --phase test \
        --how_many 163 \
        --serial_batches \
        --which_epoch $epoch

    IMG_PATH=~/benchmarking/PPT/results/test/test_${epoch}/images

    python evaluation_edited.py --image_path $IMG_PATH
done

echo "Done!"

