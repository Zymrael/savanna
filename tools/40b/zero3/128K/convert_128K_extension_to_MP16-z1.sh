#!/bin/bash

set -euo pipefail

CHECKPOINT_BASE=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K
CHECKPOINT_DIR=${CHECKPOINT_BASE}/zero3/
TAG=global_step12500
MP_SIZE=8
OUTPUT_DIR=${CHECKPOINT_BASE}/zero1/MP${MP_SIZE}
NUM_WORKERS=1

RANK_START=${1:-0}
RANK_END=${2:-$((${MP_SIZE} - 1))}

echo "RANK_START: $RANK_START, RANK_END: $RANK_END"
CMD="python convert_zero3_to_zero1.py $CHECKPOINT_DIR $OUTPUT_DIR --tag $TAG --mp_size $MP_SIZE --num_workers $NUM_WORKERS --rank_start $RANK_START --rank_end $RANK_END"
echo $CMD
$CMD

#OUTPUTS: /lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/zero1/MP8/global_step12500