#!/bin/bash

## Merge 256K zero3 checkpoint to 16 zero1 mp shards
set -euo pipefail

TAG="global_step12500" # CHANGE TO GLOBAL STEP 12500 FOR FINAL RUN

# Source MP size
MP_SIZE=64

CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-extension-n256-1M/40b_1M/202412230749
OUTPUT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/evals/40b/MP${MP_SIZE}

#Check that checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint $CHECKPOINT_DIR does not exist"
    exit 1
fi

NUM_WORKERS=1

# #Calculate mp_size - 1
RANK_START=${1:-0}
RANK_END=${2:-$((${MP_SIZE} - 1))}

echo "RANK_START: $RANK_START, RANK_END: $RANK_END"
CMD="python convert_zero3_to_zero1.py $CHECKPOINT_DIR $OUTPUT_DIR --tag $TAG --mp_size $MP_SIZE --num_workers $NUM_WORKERS --rank_start $RANK_START --rank_end $RANK_END"
echo $CMD
$CMD

echo $OUTPUT_DIR/${TAG}
ls $OUTPUT_DIR/${TAG} | wc -l
