#!/bin/bash
#SBATCH --account=nxd338_nih
#SBATCH --partition=mgc-nih
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32000MB
#SBATCH --time=240:00:00
#SBATCH --job-name=BestPose
#SBATCH --output=job_best_pose.out
#SBATCH --error=job_best_pose.err

module load miniconda/3
conda activate torch

# Your commands here
echo "Starting job $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Using $SLURM_CPUS_ON_NODE CPUs"
echo "With GPU: $CUDA_VISIBLE_DEVICES"

python find_best_pose.py --workers 16 --batch 5

