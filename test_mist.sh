#!/bin/bash -l
#SBATCH --job-name=test_ppt_mist
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_ppt_mist_cascade.out
#SBATCH --error=logs/test_ppt_mist_cascade.err

module unload python || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt_a100

echo "Using python: $(which python)"
python --version
nvidia-smi

cd ~/benchmarking/PPT
echo "Testing all epochs..."

for ckpt in checkpoints/trained_on_mist/*_net_G.pth; do
    epoch=$(basename "$ckpt" | cut -d'_' -f1)
    echo "=== Testing epoch $epoch ==="

    python test.py \
        --dataroot  ~/benchmarking/datasets/MIST_HER2/ \
        --name trained_on_mist \
        --phase test \
        --how_many 1000 \
        --serial_batches \
        --which_epoch "$epoch"

    IMG_PATH=~/benchmarking/PPT/results/trained_on_mist/test_${epoch}/images
    python evaluation_edited.py --image_path "$IMG_PATH"
done

echo "Done!"
