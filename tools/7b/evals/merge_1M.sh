#!/bin/bash
set -euo pipefail

source "utilities.sh"
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-n32-v3-hybrid-log_evo1-64K/7b-hybrid-log_evo1-64K/202411231002
ITERATION=12500
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
ROPE_SCALE="hybrid-log_evo1"

# Step 2 when extending filter length
SOURCE_SEQ_LEN_STR="1M"

NUM_GROUPS=256
SEQ_LENGTH=1048576

echo "ITERATION: ${ITERATION}, ROPE_SCALE: ${ROPE_SCALE}"

# ------------- #
echo "Step 1: Merge MP32 -> MP16"

MP_SIZE=16
SOURCE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-n32-hybrid-log_evo1-1M/7b-hybrid-log_evo1-1M/202412211305/"
SOURCE_DIR="${SOURCE_DIR}/global_step${ITERATION}"
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-evals/${SOURCE_SEQ_LEN_STR}/mp${MP_SIZE}/global_step${ITERATION}"

check_substring "$SOURCE_DIR" "$ROPE_SCALE"
check_substring "$SOURCE_DIR" "$SOURCE_SEQ_LEN_STR"
check_directory_exists "$SOURCE_DIR"
check_directory_does_not_exist "$OUTPUT_DIR"

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --mp_size $MP_SIZE"

echo $CMD
eval $CMD

ls -lth $OUTPUT_DIR
echo "Step 1: Done"
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-evals/1M/mp16/global_step12500
