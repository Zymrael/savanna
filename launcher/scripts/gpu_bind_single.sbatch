#!/bin/bash

#SBATCH --job-name=eight_gpus
#SBATCH --partition=pool0
#SBATCH --account=dir_arc
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8  # Request 8 GPUs
#SBATCH --output=logs/eight_gpus_%j.out
#SBATCH --error=logs/eight_gpus_%j.err

# srun --gres=gpu:1 --gpu-bind=map_gpu:1 nvidia-smi -L
# srun --gres=gpu:1 -n8 --ntasks-per-gpu=1 --gpu-bind=map_gpu:2 nvidia-smi -L  
srun -n1 --ntasks-per-gpu=1 --gpu-bind=map_gpu:0 nvidia-smi -L  
srun -n1 --ntasks-per-gpu=1 --gpu-bind=map_gpu:2 nvidia-smi -L  
echo "All tasks completed" 