#!/bin/bash
set -euo pipefail

ITERATION=12500
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"

# ------------- #

MP_SIZE=32
SOURCE_DIR="${CHECKPOINT_DIR}/evals/40b/MP64/global_step${ITERATION}"
OUTPUT_DIR="${CHECKPOINT_DIR}/evals/40b/MP32/global_step${ITERATION}"
ZERO_STATE="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-extension-n256-1M/40b_1M/202412230749/global_step12500/zero_pp_rank_0_mp_rank_00_model_states.pt"

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --zero3_model_state $ZERO_STATE \
    --mp_size $MP_SIZE"

echo $CMD
eval $CMD

ls -lth $OUTPUT_DIR