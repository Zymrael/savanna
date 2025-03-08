#!/bin/bash

## Merge 256K zero3 checkpoint to 16 zero1 mp shards
set -euo pipefail

TAG="global_step12500" # CHANGE TO GLOBAL STEP 12500 FOR FINAL RUN

FINAL_CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-extension-n256-256K/40b_256K/202412191045
FINAL_CHECKPOINT=${FINAL_CHECKPOINT_DIR}/${TAG}
CHECKPOINT_BASE=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved/256K
CHECKPOINT_DIR=${CHECKPOINT_BASE}/zero3
NEW_CHECKPOINT=${CHECKPOINT_DIR}/${TAG}

#Check that final checkpoint exists
# if [ ! -d "$FINAL_CHECKPOINT" ]; then
#     echo "Final checkpoint $FINAL_CHECKPOINT does not exist"
#     exit 1
# fi

# Check that new checkpoint does not exist
# if [ -d "$NEW_CHECKPOINT" ]; then
#     echo "New checkpoint $NEW_CHECKPOINT already exists"
#     exit 1
# fi

# Check 
# One time move
# mkdir -p $CHECKPOINT_DIR/$TAG
# CMD="mv $FINAL_CHECKPOINT_DIR/$TAG/ $CHECKPOINT_DIR/$TAG/"
# echo $CMD
# $CMD

# # Source MP size
MP_SIZE=16
OUTPUT_DIR=${CHECKPOINT_BASE}/zero1/MP${MP_SIZE}
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
