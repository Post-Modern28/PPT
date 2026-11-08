#!/bin/bash 
#SBATCH --job-name=eval_ppt_mist
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_ppt_mist_vlad.out
#SBATCH --error=logs/eval_ppt_mist_vlad.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi

cd ~/benchmarking/PPT
echo "Testing all epochs..."

for ckpt in checkpoints/trained_on_mist/*_net_G.pth; do
    epoch=$(basename "$ckpt" | cut -d'_' -f1)
    echo "=== Testing epoch $epoch ==="
    IMG_PATH=~/benchmarking/PPT/results/trained_on_mist/test_${epoch}/images
    python eval_by_vlad.py  --results_dir $IMG_PATH --output_csv ~/benchmarking/PPT/metrics/MIST/epoch_${epoch}.csv --batch_size 1 --num_workers 2
done

echo "Done!"

