#!/bin/bash

set -euo pipefail
MP_RANK=${1:-0}
SOURCE_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/8K/zero1/global_step516000
OUTPUT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/8K/interleaved/zero1
MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/pretrain/40b_train_8K.yml
SOURCE_MP_SIZE=8
CMD="python conversion/interleave_model_states.py --source_dir $SOURCE_DIR --output_dir $OUTPUT_DIR --model_config $MODEL_CONFIG --mp_size $SOURCE_MP_SIZE"

echo $CMD
eval $CMD