#!/bin/bash
#SBATCH --job-name=eight_gpus
#SBATCH --partition=pool0
#SBATCH --account=dir_arc
#SBATCH --nodes=1
#SBATCH --ntasks=1  # Only 1 task per array element
#SBATCH --gres=gpu:1 # Only 1 GPU per array element
#SBATCH --array=0-7%8  # Create 8 array job elements
#SBATCH --output=logs/eight_gpus_%A_%a.out
#SBATCH --error=logs/eight_gpus_%A_%a.err

srun --gres=gpu:1 --gpu-bind=map_gpu:$SLURM_ARRAY_TASK_ID nvidia-smi -L
echo "All tasks completed"