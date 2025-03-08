#!/bin/bash

set -euo pipefail

SOURCE_MP_SIZE=8 # Change to 16
TARGET_MP_SIZE=64
CONTEXT_LEN=128K # Change to 256K
ITERATION=12500 # Set to intermediate chkpt for now, make sure to update to last step 12500 for final run

#
BASE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/interleaved"
SOURCE_DIR="${BASE_DIR}/zero1/MP${SOURCE_MP_SIZE}/global_step${ITERATION}"
OUTPUT_DIR="${BASE_DIR}/zero1/MP${TARGET_MP_SIZE}/global_step${ITERATION}"
ZERO3_MODEL_STATE="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/zero3/global_step${ITERATION}/zero_pp_rank_0_mp_rank_00_model_states.pt"

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --zero3_model_state $ZERO3_MODEL_STATE \
    --mp_size $TARGET_MP_SIZE"

echo $CMD
eval $CMD
#Outputs: 