#!/bin/bash

# Shard the already extended 256K -> 1M context length mp shards to 64 mp shards
set -euo pipefail

SOURCE_MP_SIZE=16
TARGET_MP_SIZE=64

# We've already extended source context length to 1M, so we can just use that
SOURCE_CONTEXT_LEN=256K
EXTENDED_CONTEXT_LEN=1M

ITERATION=12500 # Set to intermediate chkpt for now, make sure to update to last step 12500 for final run

BASE_CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved"
BASE_DIR="${BASE_CHECKPOINT_DIR}/${EXTENDED_CONTEXT_LEN}"

# Source model shards
SOURCE_DIR="${BASE_DIR}/zero1/MP${SOURCE_MP_SIZE}/global_step${ITERATION}"
OUTPUT_DIR="${BASE_DIR}/zero1/MP${TARGET_MP_SIZE}/global_step${ITERATION}"

# Metadata for checking, not actually used in sharding params
ZERO3_MODEL_STATE="${BASE_CHECKPOINT_DIR}/${SOURCE_CONTEXT_LEN}/zero3/global_step${ITERATION}/zero_pp_rank_0_mp_rank_00_model_states.pt"

# Check that source dir exists, if not, exit
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory $SOURCE_DIR does not exist"
    exit 1
fi

#Check that output dir does not exist, if it does, exit
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR already exists"
    exit 1
fi

# Check that zero3 FILE exists, if not, exit
if [ ! -f "$ZERO3_MODEL_STATE" ]; then
    echo "Zero3 file $ZERO3_MODEL_STATE does not exist"
    exit 1
fi

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --zero3_model_state $ZERO3_MODEL_STATE \
    --mp_size $TARGET_MP_SIZE"

echo $CMD
eval $CMD

echo "Output directory $OUTPUT_DIR"
# ls $OUTPUT_DIR | wc -l

#Outputs: 