#!/bin/bash -l
#SBATCH --job-name=test_ppt_mist
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=logs/test_ppt_mist_filtered.out
#SBATCH --error=logs/test_ppt_mist_filtered.err

module unload python || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppt_a100

echo "Using python: $(which python)"
python --version
nvidia-smi

cd ~/benchmarking/PPT
echo "Testing all epochs..."




python test.py \
    --dataroot  ~/benchmarking/datasets/MIST_HER2_filtered_25 \
    --name trained_on_mist_filtered \
    --phase test \
    --how_many 1000 \
    --serial_batches \
    --which_epoch latest


echo "Done!"
