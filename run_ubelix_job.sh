#!/bin/bash
#SBATCH --job-name="IR"
#SBATCH --mail-user=ronja.stern@students.unibe.ch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu

# load modules
module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.3.0-GCC-10.2.0

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate ir

# Put your code below this line

python main.py
# python evaluate_bm25.py
# python train_sbert.py
# python evaluate_multilingual_bm25.py

# IMPORTANT:
# Run with                  sbatch run_hpc_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --cpus-per-task=8 --time=02:00:00 --pty /bin/bash