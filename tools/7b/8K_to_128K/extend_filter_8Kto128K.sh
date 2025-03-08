#!/bin/bash

set -euo pipefail

DEFAULT_DIR_BASE="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
SOURCE_DIR="$DEFAULT_DIR_BASE/7b_base_evo2_converted/interleaved/8K/MP1/global_step500000"
OUTPUT_DIR="$DEFAULT_DIR_BASE/7b_base_evo2_converted/interleaved/128K/MP1/global_step500000"
NUM_GROUPS=256
SEQ_LENGTH=8192
TARGET_LENGTH=131072

CMD="python conversion/extend_filter.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_groups $NUM_GROUPS \
    --seq_len $SEQ_LENGTH \
    --target_seq_len $TARGET_LENGTH \
    --overwrite"

echo $CMD
eval $CMD