#!/bin/bash

set -euo pipefail

MP_START=${1:-0}
MP_END=${2:-7}

CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-test-n16-32layer_zero3/32layer_zero3/202412141524
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-test-n16-32layer_zero3-8K-mp8-cp8/32layer_zero3-8K-mp8-cp8/202412141416 #./test_checkpoints/8K/ #"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-tests/4layer_zero3/"
TAG="global_step10"
OUTPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-extension-test/n16/32K/zero3/MP8CP1" # mp${MP_START}-${MP_END}"
mkdir -p $OUTPUT_DIR
MP_SIZE=8
DP_SIZE=16
NUM_WORKERS=-1

SOURCE_MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3.yml
#"/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3-8K-mp8-cp8.yml" #"/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/tests/checkpoint_loading/4layer_zero3.yml"
TARGET_MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3-8K-mp8-cp8-extended.yml
#/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3-extended.yml
#"/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3-8K-mp8-cp8-extended.yml"

CMD="python extend_zero3_checkpoint_debug.py $CHECKPOINT_DIR --output_dir $OUTPUT_DIR --mp_size $MP_SIZE --dp_size $DP_SIZE --source_model_config $SOURCE_MODEL_CONFIG --target_model_config $TARGET_MODEL_CONFIG --tag $TAG --num_workers $NUM_WORKERS --start_mp_rank $MP_START --end_mp_rank $MP_END"

echo $CMD
eval $CMD