#!/bin/bash
#SBATCH --job-name=eval_ppt
#SBATCH --partition=mem
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_ppt_mist_mem.out
#SBATCH --error=logs/eval_ppt_mist_mem.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt
nvidia-smi
cd ~/benchmarking/PPT
echo "Evaluating..."
python evaluation.py
echo "Done!"

