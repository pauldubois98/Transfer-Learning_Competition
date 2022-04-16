#!/bin/bash
#SBATCH --job-name=DL_improve
#SBATCH --output=%x.%j.txt
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --partition=cpu_long
#SBATCH --mem=16G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate intercentrale2022

# Run python script
python3.9 improve.py $SLURM_ARRAY_TASK_ID