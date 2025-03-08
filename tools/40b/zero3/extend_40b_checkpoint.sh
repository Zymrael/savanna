#!/bin/bash

set -euo pipefail

MP_START=${1:-0}
MP_END=${2:-7}
CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-8K-backup/ #"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-tests/4layer_zero3/"
TAG="global_step521000"
OUTPUT_DIR="extended_checkpoints/40b/512K/mp${MP_START}-${MP_END}"
MP_SIZE=8
DP_SIZE=256
NUM_WORKERS=-1

SOURCE_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/pretrain/40b_train_8K.yml" #"/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/tests/checkpoint_loading/4layer_zero3.yml"
TARGET_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/extension/40b_512K.yml"

CMD="python extend_zero3_checkpoint.py $CHECKPOINT_DIR --output_dir $OUTPUT_DIR --mp_size $MP_SIZE --dp_size $DP_SIZE --source_model_config $SOURCE_MODEL_CONFIG --target_model_config $TARGET_MODEL_CONFIG --tag $TAG --num_workers $NUM_WORKERS --start_mp_rank $MP_START --end_mp_rank $MP_END"

echo $CMD
eval $CMD