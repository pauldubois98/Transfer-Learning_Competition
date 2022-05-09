#!/bin/bash
#SBATCH --job-name=fid_glasses_no_glasses
#SBATCH --output=outs/%x.%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=gpup100
#SBATCH --mem=20G

# Load necessary modules
module purge
module load cuda/10.2.89/intel-19.0.3.199
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate intercentrale2022

for skip_connections_val in 0 1 2
do
	for size_val in 128 256 512
	do
		for lambda_identity_val in 1.0
		do
			for one_sided_label_smoothing_val in 0.1
			do
				for repetition_number_val in 0 1
				do
					for epoch in 1 101 201 301
						do
							echo "Comparing '/gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/data/val/glasses' with '/gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/saved_images_${repetition_number_val}/glasses_no_glasses/skip_${skip_connections_val}/${size_val}/l_identity_${lambda_identity_val}/osls_${one_sided_label_smoothing_val}/was_no_glasses/epoch_${epoch}'"
							python3.9 -m pytorch_fid /gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/data/val/glasses /gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/saved_images_${repetition_number_val}/glasses_no_glasses/skip_${skip_connections_val}/${size_val}/l_identity_${lambda_identity_val}/osls_${one_sided_label_smoothing_val}/was_no_glasses/epoch_${epoch} --device cuda:0 || echo Fail
							echo "Comparing '/gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/data/val/no_glasses' with '/gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/saved_images_${repetition_number_val}/glasses_no_glasses/skip_${skip_connections_val}/${size_val}/l_identity_${lambda_identity_val}/osls_${one_sided_label_smoothing_val}/was_glasses/epoch_${epoch}'"
							python3.9 -m pytorch_fid /gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/data/val/no_glasses /gpfs/workdir/lhotter/ChallengeIntercentrales2022/cycleGAN/saved_images_${repetition_number_val}/glasses_no_glasses/skip_${skip_connections_val}/${size_val}/l_identity_${lambda_identity_val}/osls_${one_sided_label_smoothing_val}/was_glasses/epoch_${epoch} --device cuda:0 || echo Fail
						done
				done
			done
		done
	done
done
