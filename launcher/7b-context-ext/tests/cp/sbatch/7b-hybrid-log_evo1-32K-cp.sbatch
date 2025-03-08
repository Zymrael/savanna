#!/bin/bash

#SBATCH --job-name=7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp
#SBATCH --partition=pool0
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --output=/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/logs/7b-hybrid-log_evo1-32K-cp/slurm-%N-%J.out
#SBATCH --error=/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/logs/7b-hybrid-log_evo1-32K-cp/slurm-%N-%J.err
#SBATCH --account=dir_arc
#SBATCH --dependency=singleton
#SBATCH --comment='{"APS": {}}'

set -eo pipefail

# Environment info, for debugging
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES
NTASKS=$SLURM_NTASKS
NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
echo "NNODES: $NNODES NGPUS: $GPUS_PER_NODE NTASKS: $NTASKS NTASKS_PER_NODE: $NTASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

HOSTLIST="/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/7b-hybrid-log_evo1-32K-cp-hostlist"
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $NODELIST | tr ' ' '\n' > $HOSTLIST
cat $HOSTLIST

# These are distributed env vars expected by `savanna`
# LOCAL_RANK, RANK, and WORLD_SIZE will be set by the launcher, either `torchrun` or `deepspeed.launcher.launch`
# In the case of `srun`, we need to set these explicitly, of which LOCAL_RANK and RANK are only known during `srun` execution
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
LOCAL_WORLD_SIZE=$GPUS_PER_NODE
export WORLD_SIZE
export LOCAL_WORLD_SIZE
# Note that the container defines its own WORLD_SIZE env var which container-env doesn't seem to be able to override
# We set GLOBAL_NUM_GPUS as a custom env var which the container can pick up
GLOBAL_NUM_GPUS=$WORLD_SIZE
export GLOBAL_NUM_GPUS
echo "WORLD_SIZE: $WORLD_SIZE LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"
MASTER_NODE=$(head -n 1 $HOSTLIST)
export MASTER_NODE
MASTER_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
export MASTER_NODE_IP
MASTER_PORT=$((14933 + ${SLURM_ARRAY_TASK_ID:-0}))
export MASTER_PORT
echo "MASTER_NODE: $MASTER_NODE, MASTER_NODE_IP: $MASTER_NODE_IP, MASTER_PORT: $MASTER_PORT"


# heimdall: Create the eventual srun output for the otel collector, and symlink back to launcher.
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
export APS_LOG_DIR=/home/svc-aps-wfc/otelfilelog_paths/dev
export APS_LOG_FILE=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${DATETIME}.log
touch ${APS_LOG_DIR}/${APS_LOG_FILE}
ln -sf /lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/logs/7b-hybrid-log_evo1-32K-cp/${APS_LOG_FILE} ${APS_LOG_DIR}/${APS_LOG_FILE} 
ln -sf /lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/logs/7b-hybrid-log_evo1-32K-cp/${APS_LOG_FILE} ${APS_LOG_DIR}/${APS_LOG_FILE} 
export DATA_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/data"

# for wandb logging
export WANDB_API_KEY
ROOT_DIR="/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext"
cd $ROOT_DIR

CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh"
OUTPUT_DIR="/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613"

CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs/7b-hybrid-log_evo1-32K-cp"

UPDATE_PYTHONPATH="export PYTHONPATH=$ROOT_DIR:\$PYTHONPATH"
export UPDATE_PYTHONPATH
SRUN_ARGS="-l -u --output $LOG_DIR/$APS_LOG_FILE \
--container-image $CONTAINER \
--container-workdir $ROOT_DIR \
--container-mounts $DATA_DIR:/data,$ROOT_DIR:$ROOT_DIR,$OUTPUT_DIR:$OUTPUT_DIR,$CHECKPOINT_DIR:$CHECKPOINT_DIR \
--no-container-mount-home"

# 1. Parse configs and generate train args string to pass to launcher
# Only executed on master node
# - parses and writes formatted args to $TRAIN_ARGS_OUTPUT_PATH
# - compiles data helpers for data loading
# - since all downstream training processes access same savanna root, 
# this should only be done once to prevent race conditions.
# - checks that global_num_gpus, which savanna depends on to set up parallelisms
# is correctly set based on env vars.
# We explicitly set WORLD_SIZE above and implement additional checks
# - In the case of torchrun / deepspeed, `$SLURM_NTASKS * $SLURM_JOB_NUM_NODES == $SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE`
# - In the case of `srun`, `$SLURM_NTASKS == $SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE`
DATA_CONFIG_DIR="/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/data_config"
MODEL_CONFIG_DIR="/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n4-hybrid-log_evo1-32K-cp/202411250613/model_configs"
DATA_CONFIG="$DATA_CONFIG_DIR/longphase_v3_nvidia.yml"
MODEL_CONFIG="$MODEL_CONFIG_DIR/7b-hybrid-log_evo1-32K-cp.yml"
RUN_ID="n4-hybrid-log_evo1-32K-cp"
TRAIN_ARGS_OUTPUT_PATH="$OUTPUT_DIR/n4-hybrid-log_evo1-32K-cp-train_args.txt"
TRAIN_SCRIPT="/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/train.py"
export DATA_CONFIG
export MODEL_CONFIG
export TRAIN_ARGS_OUTPUT_PATH
export TRAIN_SCRIPT

PARSER_CMD="python /lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/launcher/config_parser.py \
/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/train.py \
$DATA_CONFIG \
$MODEL_CONFIG \
--hostlist $HOSTLIST \
--train-args-output $TRAIN_ARGS_OUTPUT_PATH  --wandb_project 7b-context-extension-cp-test --wandb_group 32K --wandb_run_name $RUN_ID"
PARSER_CMD="srun $SRUN_ARGS --nodes 1 --ntasks 1 -w $MASTER_NODE bash -c 'export PYTHONPATH=$ROOT_DIR:\$PYTHONPATH && make -C $ROOT_DIR/savanna/data && ${PARSER_CMD}'"

echo $PARSER_CMD
eval $PARSER_CMD

# 2. Launcher
# Runs `srun` launches world_size number of processes
# We need to set RANK and LOCAL_RANK manually
NSYS_CMD=""
PROGRAM_CMD="LOCAL_RANK=\$SLURM_LOCALID RANK=\$SLURM_PROCID python $TRAIN_SCRIPT \$(<$TRAIN_ARGS_OUTPUT_PATH)"
export NSYS_CMD
export PROGRAM_CMD
LAUNCHER_CMD="$NSYS_CMD $PROGRAM_CMD"
CMD="srun $SRUN_ARGS bash -c \
'echo \$SLURM_PROCID \$SLURM_LOCALID $SLURM_PROCID $SLURM_LOCALID && \
LOCALID=\$SLURM_LOCALID && \
if (( LOCALID == 0 )); then \
pip install arrow ring_flash_attn; \
fi;'"
#  && \
# export TRITON_CACHE_DIR=/tmp/triton_cache-\$SLURM_JOB_ID-\$SLURM_STEP_ID-\$SLURM_PROCID-\$SLURM_NODEID && mkdir -p \$TRITON_CACHE_DIR; \
# export PYTHONPATH=$ROOT_DIR:$PYTHONPATH && ${LAUNCHER_CMD}'"
# # 'if (( \$SLURM_LOCALID == 0 )); then \
#     echo "\$SLURM_PROCID \$SLURM_LOCALID" && \
#     pip install -y --force-reinstall tokenizers==4.45.0 arrow ring_flash_attn; \
# fi && \
# export TRITON_CACHE_DIR=/tmp/triton_cache-\$SLURM_JOB_ID-\$SLURM_STEP_ID-\$SLURM_PROCID-\$SLURM_NODEID && mkdir -p \$TRITON_CACHE_DIR; \
# export PYTHONPATH=$ROOT_DIR:$PYTHONPATH && ${LAUNCHER_CMD}'"

echo $CMD
eval $CMD
