#!/bin/bash
set -euo pipefail

# This step occurs after merging zero3 checkpoint to source mp shards (16), before we have resharded to target mp shards (64)
SOURCE_MP_SIZE=16
TARGET_MP_SIZE=$SOURCE_MP_SIZE

# Extend filter lengths to 1M
SOURCE_CONTEXT_LEN=256K 
TARGET_CONTEXT_LEN=1M

echo "Step 1: Extend ${SOURCE_CONTEXT_LEN} -> ${TARGET_CONTEXT_LEN}"

ITERATION=12500 #12500 # Set to intermediate chkpt for now, make sure to update to last step 12500 for final run

SOURCE_BASE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved/${SOURCE_CONTEXT_LEN}"
TARGET_BASE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved/${TARGET_CONTEXT_LEN}"
SOURCE_DIR="${SOURCE_BASE_DIR}/zero1/MP${SOURCE_MP_SIZE}/global_step${ITERATION}"
OUTPUT_DIR="${TARGET_BASE_DIR}/zero1/MP${TARGET_MP_SIZE}/global_step${ITERATION}"

SOURCE_MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-1M/configs/40b/model_configs/extension/256K/40b_256K.yml
TARGET_MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-1M/configs/40b/model_configs/extension/1M/40b_1M.yml

# Check that source dir exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Check that output dir does not exist, if it does, exit
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR already exists"
    exit 1
fi

#Check that source and target model configs exist
if [ ! -f "$SOURCE_MODEL_CONFIG" ]; then
    echo "Source model config $SOURCE_MODEL_CONFIG does not exist"
    exit 1
fi

if [ ! -f "$TARGET_MODEL_CONFIG" ]; then
    echo "Target model config $TARGET_MODEL_CONFIG does not exist"
    exit 1
fi

# CMD="python conversion/extend_filter.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --source_model_config $SOURCE_MODEL_CONFIG \
#     --target_model_config $TARGET_MODEL_CONFIG"

# echo $CMD
# eval $CMD

ls -lth $OUTPUT_DIR
echo "Step 2: Done"


# echo "Step 3: Check ${TARGET_LEN_STR} filter"

# SOURCE_DIR=$OUTPUT_DIR

# CMD="python checks/check_filter_lens.py \
#     --source_dir $SOURCE_DIR \
#     --num_groups $NUM_GROUPS \
#     --seq_len $SEQ_LENGTH \
#     --target_seq_len $TARGET_LENGTH"

# echo $CMD
# eval $CMD

#echo "Step 3: Done"
