#!/bin/bash
set -euo pipefail

TARGET_MP_SIZE=64
TARGET_DP_SIZE=32 # Change to 32 for 2048 GPU run
TARGET_CONTEXT_LEN="1M"
TARGET_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-1M/configs/40b/model_configs/extension/1M/40b_1M.yml"

CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/interleaved/${TARGET_CONTEXT_LEN}/zero3/MP${TARGET_MP_SIZE}DP${TARGET_DP_SIZE}/padded/global_step0"
CMD="python extension-checks/check_padding_and_extended_filters.py --checkpoint_dir $CHECKPOINT_DIR --target_model_config ${TARGET_MODEL_CONFIG}"

echo $CMD
eval $CMD