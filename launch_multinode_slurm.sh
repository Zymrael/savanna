#!/bin/bash

#SBATCH --nodes=4
#SBATCH --partition=gpu_batch
#SBATCH --job-name=40b_test_config
#SBATCH --output=log/%j_40b_test_config.log
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

GPUS_PER_NODE=4

scontrol show hostname ${SLURM_JOB_NODELIST} > hostfile
sed -i "s/$/ slots=${GPUS_PER_NODE}/" hostfile

export SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

MASTER_NODE=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
CURR_NODE=$(hostname)
if [ "$CURR_NODE" = "$MASTER_NODE" ]; then
    while true
    do
        cat hostfile

        python launch.py train.py -d configs data/opengenome.yml model/evo2/40b_test_config.yml

        sleep 900
    done
fi
