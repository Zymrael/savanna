#!/bin/bash

set -euo pipefail

SOURCE_MP_SIZE=8
TARGET_MP_SIZE=16
CONTEXT_LEN=128K #512K
ITERATION=12500

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
#Outputs: /lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero1/MP16/global_step12500