#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_batch
#SBATCH --job-name=test_ft
#SBATCH --cpus-per-task=4
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --signal=B:SIGINT@80
#SBATCH --output=inference_log.txt
#SBATCH --open-mode=append

GPUS_PER_NODE=2

# Get the names of all nodes involved in training.
scontrol show hostname ${SLURM_JOB_NODELIST} > 7b_test
sed -i "s/$/ slots=${GPUS_PER_NODE}/" 7b_test

# Launch the main traning run on only the master node.
MASTER_NODE=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
CURR_NODE=$(hostname)
if [ "$CURR_NODE" = "$MASTER_NODE" ]; then
     python ./launch.py inference.py -d configs data/opengenome2.yml model/evo2/7b_13h_8m_8s_3a_cascade15_test_loss.yml
fi
