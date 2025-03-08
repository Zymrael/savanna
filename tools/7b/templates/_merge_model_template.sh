#!/bin/bash

#  cat ../../7b-context-extension-n32-evo1-32K/202411100226/model_configs/7b-evo1-32K.yml | grep -A2 save
set -euo pipefail

ITERATION=12500
SOURCE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-n32-evo1-32K/7b-evo1-32K/202411100226/global_step${ITERATION}"
OUTPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension/evo1/32K/MP1/global_step${ITERATION}"
MP_SIZE=1

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --mp_size $MP_SIZE"

echo $CMD
# eval $CMD
