#!/bin/bash

set -euo pipefail

# Only for reference, as the model state / optim state files of rank0 will be used, mp_size * dp_size should be 2048
SOURCE_MP_SIZE=16
SOURCE_DP_SIZE=128

#Check that mp_size * dp_size is 2048
if [ $(($SOURCE_MP_SIZE * $SOURCE_DP_SIZE)) -ne 2048 ]; then
    echo "Source MP size * DP size is not 2048"
    exit 1
fi

# These are the actual MP/DP sharded partition sizes
TARGET_MP_SIZE=64
TARGET_DP_SIZE=32 #Change to 32 for 2048 GPU run; 16 for 1024 GPU test; 8 for 512 GPU test; 4 for 256 GPU test; 2 for 128 GPU test

SOURCE_CONTEXT_LEN="256K" # Need original model state and optim state files for metadata
TARGET_CONTEXT_LEN="1M" # Extended, resharded zero1 shards
TARGET_MODEL_CONFIG_PATH="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-1M/configs/40b/model_configs/extension/1M/40b_1M.yml"

SOURCE_TAG="global_step12500" #Change to 12500 for final run
OUTPUT_TAG="global_step0"

BASE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved"
SOURCE_ZERO3_DIR="${BASE_DIR}/${SOURCE_CONTEXT_LEN}/zero3/${SOURCE_TAG}"
SHARDED_ZERO1_DIR="${BASE_DIR}/${TARGET_CONTEXT_LEN}/zero1/MP${TARGET_MP_SIZE}/${SOURCE_TAG}"
OUTPUT_DIR="${BASE_DIR}/${TARGET_CONTEXT_LEN}/zero3/MP${TARGET_MP_SIZE}DP${TARGET_DP_SIZE}/padded/${OUTPUT_TAG}"

# Check that source dir exists
if [ ! -d "$SOURCE_ZERO3_DIR" ]; then
    echo "Source directory $SOURCE_ZERO3_DIR does not exist"
    exit 1
fi

# Check that sharded zero1 dir exists
if [ ! -d "$SHARDED_ZERO1_DIR" ]; then
    echo "Sharded zero1 directory $SHARDED_ZERO1_DIR does not exist"
    exit 1
fi

# # Check that output dir does not exist
# if [ -d "$OUTPUT_DIR" ]; then
#     echo "Output directory $OUTPUT_DIR already exists"
#     exit 1
# fi

# Script processes range of ranks -- INCLUSIVE, 0 2 -> 0, 1, 2
START_MP_RANK=${1:-0}
END_MP_RANK=${2:-$((${TARGET_MP_SIZE} - 1))}
NUM_WORKERS=$((${END_MP_RANK}+1))

CMD="python partition_param.py --source_mp_size $SOURCE_MP_SIZE --source_dp_size $SOURCE_DP_SIZE --target_mp_size $TARGET_MP_SIZE \
--target_dp_size $TARGET_DP_SIZE \
--source_zero3_dir $SOURCE_ZERO3_DIR \
--sharded_zero1_dir $SHARDED_ZERO1_DIR \
--output_dir $OUTPUT_DIR \
--num_workers $NUM_WORKERS \
--pad_mlp_weights \
--target_model_config_path $TARGET_MODEL_CONFIG_PATH \
--start_mp_rank $START_MP_RANK \
--end_mp_rank $END_MP_RANK"

echo $CMD
eval $CMD

echo "Output directory $OUTPUT_DIR"
# ls $OUTPUT_DIR | wc -l