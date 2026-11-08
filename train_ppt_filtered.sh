#!/bin/bash
#SBATCH --job-name=train_ppt
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_ppt_filtered_%j.out
#SBATCH --error=logs/train_ppt_filtered_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt_a100
nvidia-smi
cd ~/benchmarking/PPT
echo "Training..."
python train.py --dataroot ~/benchmarking/datasets/MIST_HER2_filtered_25/ --name trained_on_mist_filtered --save_epoch_freq 5 --epoch_count 54 --continue_train
echo "Done!"

