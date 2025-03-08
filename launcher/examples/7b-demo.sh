# Starter example of how to use `generate_distributed_launcher.py` for a 7B model
#
# Usage:
# From `launcher` directory
# .examples/7b-demo.sh
# This will output: 
# /path/to/logs
# /path/to/scripts
# ...
# Submit and check logs: `sbatch .../$JOB_NAME/.../scripts/7b.xxx.sbatch`

SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
# MODEL_CONFIGS regression_test, 7b_shc_post_refactor-mp2-dp2, 7b_shc_post_refactor-dp4, 40b_test_config.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/7b_shc_post_refactor-mp-dp.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
OLD_CONTAINER=nvidia_evo2_efa_latest.sqsh
LATEST_HEIMDALL=clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-22.sqsh
LATEST_NO_HEIMDALL=clara-discovery+savanna+arc-evo2_efa+pt24.09-py3_ncclv2.23.4-2024-10-21.sqsh
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/$LATEST_HEIMDALL

NUM_NODES=2
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

# Change to your desired launcher: torch, deepspeed, srun
# They are functionally equivalent, srun is the default
# deepspeed has the additional feature of --enable-each-rank-log, which outputs per-rank logs to rank_logs directory
# Best to use `srun` for now, as we will be integrating more performance-oriented features only available with `srun`
LAUNCHER=torch
OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME

# Additional flags that will override user-provided model config
# --use_wandb: wandb monitoring -- make sure to export your WANDB_API_KEY if using wandb, otherwise will get assertion error
# needed to use wandb within container
# --nsys: launches nsys profiling; see `python generated_distributed_launcher.py --help` for all available nsys options 
CMD="python generate_distributed_launcher.py \
$JOB_NAME \
--use-wandb \
--output-dir $OUTPUT_DIR \
--launcher $LAUNCHER \
--partition $PARTITION \
--account $ACCOUNT \
--container $CONTAINER \
--num-nodes $NUM_NODES \
--num-gpus $NUM_GPUS \
--data-config $DATA_CONFIG \
--model-config $MODEL_CONFIG \
--train-script $TRAIN_SCRIPT"

echo $CMD
eval $CMD