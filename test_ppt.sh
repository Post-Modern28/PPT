#!/bin/bash
#SBATCH --job-name=test_ppt
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/test_ppt_%j.out
#SBATCH --error=logs/test_ppt_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt
nvidia-smi
cd ~/benchmarking/PPT
echo "Testing..."
python test.py --dataroot ~/benchmarking/datasets/MIST_HER2/ --name test --phase test --how_many 1000 --serial_batches --which_epoch latest
python evaluation.py
echo "Done!"
