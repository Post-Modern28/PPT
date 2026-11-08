#!/bin/bash 
#SBATCH --job-name=eval_ppt_mist
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_ppt_filtered_q1.out
#SBATCH --error=logs/eval_ppt_filtered_q1.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi

cd ~/benchmarking/PPT
echo "Evaluating all epochs..."

for ckpt in checkpoints/trained_on_mist_filtered/*_net_G.pth; do
    epoch=$(basename "$ckpt" | cut -d'_' -f1)
    echo "=== Testing epoch $epoch ==="
    IMG_PATH=~/benchmarking/PPT/results/trained_on_mist_filtered/test_${epoch}
    python eval_extended.py  --results_dir ${IMG_PATH}/images --output_csv ~/benchmarking/PPT/metrics/MIST/filtered/epoch_${epoch}.csv --batch_size 4 --num_workers 2
    python eval_extended.py  --results_dir ${IMG_PATH}/good --output_csv ~/benchmarking/PPT/metrics/MIST/filtered/epoch_${epoch}_good.csv --batch_size 4 --num_workers 2
    python eval_extended.py  --results_dir ${IMG_PATH}/norm --output_csv ~/benchmarking/PPT/metrics/MIST/filtered/epoch_${epoch}_norm.csv --batch_size 4 --num_workers 2
    python eval_extended.py  --results_dir ${IMG_PATH}/comb --output_csv ~/benchmarking/PPT/metrics/MIST/filtered/epoch_${epoch}_comb.csv --batch_size 4 --num_workers 2
done

echo "Done!"

