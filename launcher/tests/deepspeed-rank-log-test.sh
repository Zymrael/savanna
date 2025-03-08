# Generates SLURM batch script that runs pipeline the 7b model config
#
# Usage:
# From `launcher` directory
# .examples/7b.train.xxx.sh
# This will output: "SLURM script generated: .../$JOB_NAME/.../7b.xxx.sbatch"
# Submit and check logs: `sbatch .../$JOB_NAME/.../7b.xxx.sbatch`

#NOTE: if using config as is, only the master rank will run without error since the other ranks will attempt
# to save checkpoint, which I've explicitly disabled in the config.

# NOTE: use full path; relative paths won't be mapped correctly in the SLURM script or container
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$(realpath ./tests/check_ds_rank_log.py)
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
# MODEL_CONFIGS regression_test, 7b_shc_post_refactor-mp2-dp2, 7b_shc_post_refactor-dp4, 40b_test_config.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/deepspeed-rank-log.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh"
NUM_NODES=2
NUM_GPUS=2
PARTITION=pool0
ACCOUNT=dir_arc

LAUNCHER=deepspeed
OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME

MEM="10G"
CPUS_PER_TASK="4"

CMD="python generate_distributed_launcher.py \
$JOB_NAME \
--launcher $LAUNCHER \
--partition $PARTITION \
--account $ACCOUNT \
--container $CONTAINER \
--num-nodes $NUM_NODES \
--num-gpus $NUM_GPUS \
--mem $MEM \
--cpus-per-task $CPUS_PER_TASK \
--output-dir $OUTPUT_DIR \
--data-config $DATA_CONFIG \
--model-config $MODEL_CONFIG \
--train-script $TRAIN_SCRIPT"

echo $CMD
eval $CMD