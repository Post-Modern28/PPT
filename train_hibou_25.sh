#!/bin/bash
#SBATCH --job-name=ppt_h25t
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_ppt_hibou_25.out
#SBATCH --error=logs/train_ppt_hibou_25.err


unset PYTHONHOME
unset PYTHONPATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt_a100
nvidia-smi
cd ~/benchmarking/PPT
echo "Training..."
python train.py --dataroot ~/benchmarking/datasets/MIST_HER2_HIBOU_L_SPEC_MPP_PATCHES_25/ --name hibou_25 --save_epoch_freq 5 
echo "Done!"

