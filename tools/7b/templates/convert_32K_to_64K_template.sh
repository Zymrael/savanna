#!/bin/bash
set -euo pipefail

#  cat ../../7b-context-extension-n32-evo1-32K/202411100226/model_configs/7b-evo1-32K.yml | grep -A2 save
echo "This is a template only, set the initial source directory, ITERATION, ROPE_SCALE, SEQ / TARGET LENGTHS, and MP SIZES before continuing"
echo $STOP

ITERATION=12500
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
ROPE_SCALE=

source "utilities.sh"

# ------------- #
echo "Step 1: Merge MP2 -> MP1"

MP_SIZE=1
SOURCE_DIR=
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension/${ROPE_SCALE}/32K/MP${MP_SIZE}/global_step${ITERATION}"

check_substring "$SOURCE_DIR" "$ROPE_SCALE"

# CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --mp_size $MP_SIZE"

# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR

# # ------------- #

echo "Step 2: Extend 32K -> 64K"

NUM_GROUPS=256
SEQ_LENGTH=32768
TARGET_LENGTH=65536

SOURCE_DIR=$OUTPUT_DIR
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension/${ROPE_SCALE}/64K/MP1/global_step${ITERATION}"

# CMD="python conversion/extend_filter.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --num_groups $NUM_GROUPS \
#     --seq_len $SEQ_LENGTH \
#     --target_seq_len $TARGET_LENGTH"

# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR

# ------------- #

echo "Step 3: Check 64K filter"

SOURCE_DIR=$OUTPUT_DIR

# CMD="python checks/check_filter_lens.py \
#     --source_dir $SOURCE_DIR \
#     --num_groups $NUM_GROUPS \
#     --seq_len $SEQ_LENGTH \
#     --target_seq_len $TARGET_LENGTH"

# echo $CMD
# eval $CMD

# # ------------- #

echo "Step 4: Convert MP1 -> MP8"

MP_SIZE=8
SOURCE_DIR=$OUTPUT_DIR
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension/${ROPE_SCALE}/64K/MP${MP_SIZE}/global_step${ITERATION}"

# CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --mp_size $MP_SIZE"

# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR
