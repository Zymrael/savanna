#!/bin/bash
set -euo pipefail
TAG="global_step12500"
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved/1M/zero1/MP16/${TAG}"
SOURCE_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-1M/configs/40b/model_configs/extension/256K/40b_256K.yml"
TARGET_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-1M/configs/40b/model_configs/extension/1M/40b_1M.yml"

#Check that source dir exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Source directory $CHECKPOINT_DIR does not exist"
    exit 1
fi

# Check that source and target model configs exist
if [ ! -f "$SOURCE_MODEL_CONFIG" ]; then
    echo "Source model config $SOURCE_MODEL_CONFIG does not exist"
    exit 1
fi

if [ ! -f "$TARGET_MODEL_CONFIG" ]; then
    echo "Target model config $TARGET_MODEL_CONFIG does not exist"
    exit 1
fi

# # Check after extending MP16 shards
# CMD="python extension-checks/check_extended_filter_shards.py --source_dir $CHECKPOINT_DIR --source_model_config $SOURCE_MODEL_CONFIG --target_model_config $TARGET_MODEL_CONFIG"

# echo $CMD
# eval $CMD

# Check after repartitioning MP16 shards to MP64
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved/1M/zero1/MP64/${TAG}"

# Check that source dir exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Source directory $CHECKPOINT_DIR does not exist"
    exit 1
fi

# Change source model config to target model config to account for new model parallel size
CMD="python extension-checks/check_extended_filter_shards.py --source_dir $CHECKPOINT_DIR --source_model_config $TARGET_MODEL_CONFIG --target_model_config $TARGET_MODEL_CONFIG"

echo $CMD
eval $CMD
