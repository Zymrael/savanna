# Search config demo
# Create multiple scripts based on a template model config and a search config
#
# the search config is a yaml file with same format as the model config
# except scalars can be replaced by lists
# See search_configs/demo_search_config.yml for an example

# Usage:
# From `launcher` directory
# .examples/7b-search-config-demo.sh
# Check output for search_configs directory, which contains the generated template with each combination
# of search config values
# E.g., search_configs/config_0__train_micro_batch_size_per_gpu=1.yml, search_configs/config_1__train_micro_batch_size_per_gpu=2.yml, etc.

SAVANNA_ROOT=$(realpath ..)

# SLURM parameters
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh"
NUM_NODES=2
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc
LAUNCHER=srun

# Train params
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/7b_shc_post_refactor-mp-dp.yml
SEARCH_CONFIG=$SAVANNA_ROOT/launcher/search_configs/demo_search_config.yml
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
--search-config $SEARCH_CONFIG \
--train-script $TRAIN_SCRIPT"

echo $CMD
eval $CMD