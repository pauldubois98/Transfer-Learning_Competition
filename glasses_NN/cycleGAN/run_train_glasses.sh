#!/bin/bash
#SBATCH --job-name=glasses_no_glasses
#SBATCH --output=outs/%x.%j.txt
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem=20G

# Load necessary modules
module purge
module load cuda/10.2.89/intel-19.0.3.199
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate intercentrale2022

# Array declaration, probably should be done more cleverly once I increase the number of combinations ...
skip_connections=(0 1 2)
size=(128 128 128)
lambda_identity=(1 1 1)
one_sided_label_smoothing=(0.1 0.1 0.1)


# Prints
echo This is task "$SLURM_ARRAY_TASK_ID"
echo skip_connections = "${skip_connections[$SLURM_ARRAY_TASK_ID]}"
echo size = "${size[$SLURM_ARRAY_TASK_ID]}"
echo lambda_identity = "${lambda_identity[$SLURM_ARRAY_TASK_ID]}"
echo one_sided_label_smoothing = "${one_sided_label_smoothing[$SLURM_ARRAY_TASK_ID]}"

# Run python script
python3.9 train.py glasses no_glasses "${skip_connections[$SLURM_ARRAY_TASK_ID]}" "${size[$SLURM_ARRAY_TASK_ID]}" "${lambda_identity[$SLURM_ARRAY_TASK_ID]}" "${one_sided_label_smoothing[$SLURM_ARRAY_TASK_ID]}"
