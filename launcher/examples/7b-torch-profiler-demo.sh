# Demonstrates usage with torch.profiler
#
# Check output directory for torchprofiler_traces folder for per-rank traces
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
# Note: model config must specify should_profile=True and profiler_type="torch"
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/7b_shc_post_refactor-torch-profiler.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh"
NUM_NODES=2
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc
LAUNCHER=torch
OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME

CMD="python generate_distributed_launcher.py \
$JOB_NAME \
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