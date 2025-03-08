#!/bin/bash

set -euo pipefail

CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
ITERATION=500000
MP_SIZE=8
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
SOURCE_DIR="${CHECKPOINT_DIR}/7b_base_evo2_converted/interleaved/128K/MP1/global_step${ITERATION}"
OUTPUT_DIR="${CHECKPOINT_DIR}/7b_base_evo2_converted/interleaved/128K/MP${MP_SIZE}/extra_states/global_step${ITERATION}"

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --mp_size $MP_SIZE
    --include-extra"

echo $CMD
eval $CMD