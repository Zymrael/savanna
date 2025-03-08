#!/bin/bash

#SBATCH --job-name=eight_gpus
#SBATCH --partition=pool0
#SBATCH --account=dir_arc
#SBATCH --nodes=1
#SBATCH --ntasks=8          # Request 8 tasks
#SBATCH --gres=gpu:8       # Request 8 GPUs
#SBATCH --output=logs/gpu_bind_%j_%t.out  # %t is the task ID
#SBATCH --error=logs/gpu_bind_%j_%t.err
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

srun --output=logs/srun/gpu_bind_%J-%t.out --error=logs/srun/gpu_bind_%J-%t.err --container-image=$CONTAINER --gres=gpu:1 bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; echo $CUDA_VISIBLE_DEVICES && echo $SLURM_LOCALID && python -c "import torch; print(torch.cuda.current_device())"' #&  # Background is important!
#--gpu-bind=map_gpu:$SLURM_PROCID
wait  # Wait for all background tasks to complete

echo "All tasks completed"


# #!/bin/bash

# #SBATCH --job-name=eight_gpus
# #SBATCH --partition=pool0
# #SBATCH --account=dir_arc
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --gres=gpu:8  # Request 8 GPUs
# #SBATCH --output=logs/gpu_bind_%j.out
# #SBATCH --error=logs/gpu_bind_%j.err

# # Array indices for argument chunking
# # start_indices=(0 2 4 6 8 10 12 14)
# # end_indices=(2 4 6 8 10 12 14 16) # Example: Adjust as needed


# # Loop through tasks (0 to 7)
# for i in $(seq 0 7); do
#   # Calculate the GPU ID for this task
#     gpu_id=$i
#     echo "GPU ID: $gpu_id"
#     srun --gres=gpu:1 --gpu-bind=map_gpu:$gpu_id nvidia-smi -L
#   done
#   wait

#   # Calculate the start and end indices for the argument chunk
# #    start=${start_indices[$i]}
#  # end=${end_indices[$i]}

#   # Execute the program with srun, assigning a specific GPU
#   # your_program  ${start} ${end}  &  # Run in background
# # done

# # Wait for all background processes to complete
# # wait

# echo "All tasks completed" 