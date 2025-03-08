#!/bin/bash

set -euo pipefail

CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
SOURCE_DIR="$CHECKPOINT_DIR/7b-conversion-tests/32K/MP1/global_step500000"
NUM_GROUPS=256
SEQ_LEN=8192
TARGET_SEQ_LEN=32768

CMD="python checks/check_filter_lens.py \
    --source_dir $SOURCE_DIR \
    --num_groups $NUM_GROUPS \
    --seq_len $SEQ_LEN \
    --target_seq_len $TARGET_SEQ_LEN"

echo $CMD
eval $CMD