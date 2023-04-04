#!/bin/bash
#SBATCH --job-name="IR"
#SBATCH --mail-user=ronja.stern@students.unibe.ch
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=1G


# load modules
module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.3.0-GCC-10.2.0

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate ir

# Put your code below this line

python main.py
