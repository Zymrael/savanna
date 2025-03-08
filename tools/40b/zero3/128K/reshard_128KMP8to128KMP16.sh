#!/bin/bash

set -euo pipefail

#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-extension-n256-128K_no_recycle_avoid_streams/40b_128K_no_rc/202412160857/global_step12500/
CHECKPOINT_BASE=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K
#repartitioned_checkpoints/8K/MP8DP4
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/4layer/
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/256K
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n128-extension/8K
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-extension-n128-40b_8K_base-test/40b_8K_base/202412150045
CHECKPOINT_DIR=${CHECKPOINT_BASE}/zero3/
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/512K/zero3/
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/32K/zero3/
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-8K-backup 
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-512K/zero3/mp6-7/
# TAG=global_step0
TAG=global_step12500
MP_SIZE=16
OUTPUT_DIR=${CHECKPOINT_BASE}/zero1/MP${MP_SIZE}

NUM_WORKERS=1

#Calculate mp_size - 1
RANK_START=${1:-0}
RANK_END=${2:-$((${MP_SIZE} - 1))}

echo "RANK_START: $RANK_START, RANK_END: $RANK_END"
CMD="python convert_zero3_to_zero1.py $CHECKPOINT_DIR $OUTPUT_DIR --tag $TAG --mp_size $MP_SIZE --num_workers $NUM_WORKERS --rank_start $RANK_START --rank_end $RANK_END"
echo $CMD
$CMD