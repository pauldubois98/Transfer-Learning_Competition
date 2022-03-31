#!/bin/bash
#SBATCH --job-name=mnist_optuna
#SBATCH --output=%x.%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpup100
#SBATCH --mem=16G
#SBATCH --mail-user=lhotteromain@gmail.com
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load cuda/10.2.89/intel-19.0.3.199
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate intercentrale2022

# Run python script
python3.9 pytorch-mnist-GAN-optuna.py