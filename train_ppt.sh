#!/bin/bash
#SBATCH --job-name=train_ppt
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_ppt_%j.out
#SBATCH --error=logs/train_ppt_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt_a100
nvidia-smi
cd ~/benchmarking/PPT
echo "Training..."
python train.py --dataroot ~/benchmarking/datasets/MIST_HER2/ --name trained_on_mist --save_epoch_freq 5 --epoch_count 166 --continue_train
echo "Done!"

