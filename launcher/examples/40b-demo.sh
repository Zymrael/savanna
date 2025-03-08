# Generates SLURM batch script that runs pipeline with train script with the 40b model config
#
# Usage:
# From `launcher` directory
# ./examples/40b-demo.sh
# This will output: "SLURM script generated: .../$JOB_NAME/.../40b_test_config.sbatch"
# Submit and check logs: `sbatch .../$JOB_NAME/.../40b_test_config.sbatch`

SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/40b_test_config.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh" #/lustre/fs01/portfolios/dir/project/dir_arc/containers/clara-discovery+savanna+arc-evo2_efa+nv-latest-cascade-1.5.sqsh"
NUM_NODES=4
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

# Change to your desired launcher: torch, deepspeed, srun
# They are functionally equivalent, srun is the default
# deepspeed has the additional feature of --enable-each-rank-log, which outputs per-rank logs to rank_logs directory
LAUNCHER=srun
OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME

# Adjust as needed
JOBTIME="02:00:00"
NUM_ITERS=1000
OVERRIDES="train-iters=$NUM_ITERS \
lr-decay-iters=$NUM_ITERS"

CMD="python generate_distributed_launcher.py \
$JOB_NAME \
--job-time $JOBTIME \
--overrides $OVERRIDES \
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