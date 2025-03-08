#!/bin/bash
set -euo pipefail

source "utilities.sh"

# Extend interleaved 7b pretrained at 500k iterations to 32K
ITERATION=500000
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
ROPE_SCALE="hybrid-log_evo1"

# Step 2 when extending filter length
PREVIOUS_SEQ_LEN_STR="8K"
TARGET_LEN_STR="32K"

NUM_GROUPS=256
SEQ_LENGTH=8192
TARGET_LENGTH=32768

echo "ITERATION: ${ITERATION}, ROPE_SCALE: ${ROPE_SCALE}"

# ------------- #
# Extend filter length *in-place*

echo "Step 1: Extend ${PREVIOUS_SEQ_LEN_STR} -> ${TARGET_LEN_STR}"

SOURCE_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b_base_evo2_converted/interleaved/8K/MP1/global_step500000
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension-cp/${ROPE_SCALE}/${TARGET_LEN_STR}/MP1/global_step${ITERATION}"

# check_substring "$OUTPUT_DIR" "$ROPE_SCALE"
# check_substring "$OUTPUT_DIR" "$TARGET_LEN_STR"
# check_directory_does_not_exist "$OUTPUT_DIR"

# CMD="python conversion/extend_filter.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --num_groups $NUM_GROUPS \
#     --seq_len $SEQ_LENGTH \
#     --target_seq_len $TARGET_LENGTH"

# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR
# echo "Step 2: Done"

# ------------- #

echo "Step 3: Check ${TARGET_LEN_STR} filter"

SOURCE_DIR=$OUTPUT_DIR

CMD="python checks/check_filter_lens.py \
    --source_dir $SOURCE_DIR \
    --num_groups $NUM_GROUPS \
    --seq_len $SEQ_LENGTH \
    --target_seq_len $TARGET_LENGTH"

echo $CMD
eval $CMD

echo "Step 3: Done"

# # # ------------- #

# echo "Step 4: Convert MP1 -> MP8"

# MP_SIZE=8
# SOURCE_DIR=$OUTPUT_DIR
# OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension-v3/${ROPE_SCALE}/${TARGET_LEN_STR}/MP${MP_SIZE}/global_step${ITERATION}"

# check_substring "$OUTPUT_DIR" "$ROPE_SCALE"
# check_substring "$OUTPUT_DIR" "$TARGET_LEN_STR"
# check_substring "$OUTPUT_DIR" "MP${MP_SIZE}"
# check_directory_does_not_exist "$OUTPUT_DIR"

# CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --mp_size $MP_SIZE"

# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR
# echo "Step 4: Done"