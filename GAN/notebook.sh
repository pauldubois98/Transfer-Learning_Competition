#!/bin/bash
#SBATCH --job-name=notebook
#SBATCH --output=%x.%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --partition=gpup100
#SBATCH --mem=16G

module load anaconda3/2021.05/gcc-9.2.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate

set -x
jupyter notebook --no-browser --port=8888 --ip=$(hostname -s)

