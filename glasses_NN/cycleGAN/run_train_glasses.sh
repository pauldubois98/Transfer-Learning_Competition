#!/bin/bash
#SBATCH --job-name=glasses_no_glasses
#SBATCH --output=outs/%x.%j.txt
#SBATCH --array=0-11
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
skip_connections=()
size=()
lambda_identity=()
one_sided_label_smoothing=()
repetition_number=()
for skip_connections_val in 0 1 2
do
	for size_val in 128 256
	do
		for lambda_identity_val in 1
		do
			for one_sided_label_smoothing_val in 0.1
			do
				for repetition_number_val in 0 1
				do
					skip_connections+=($skip_connections_val)
					size+=($size_val)
					lambda_identity+=($lambda_identity_val)
					one_sided_label_smoothing+=($one_sided_label_smoothing_val)
					repetition_number+=($repetition_number_val)
				done
			done
		done
	done
done



# Prints
echo This is task "${SLURM_ARRAY_TASK_ID}"
echo skip_connections = "${skip_connections[$SLURM_ARRAY_TASK_ID]}"
echo size = "${size[$SLURM_ARRAY_TASK_ID]}"
echo lambda_identity = "${lambda_identity[$SLURM_ARRAY_TASK_ID]}"
echo one_sided_label_smoothing = "${one_sided_label_smoothing[$SLURM_ARRAY_TASK_ID]}"

# Run python script
python3.9 train.py glasses no_glasses "${skip_connections[$SLURM_ARRAY_TASK_ID]}" "${size[$SLURM_ARRAY_TASK_ID]}" "${lambda_identity[$SLURM_ARRAY_TASK_ID]}" "${one_sided_label_smoothing[$SLURM_ARRAY_TASK_ID]}" "${repetition_number[$SLURM_ARRAY_TASK_ID]}"
