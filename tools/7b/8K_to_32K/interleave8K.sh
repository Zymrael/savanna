#!/bin/bash

set -euo pipefail

DEFAULT_DIR_BASE="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
SOURCE_CHECKPOINT="$DEFAULT_DIR_BASE/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/global_step500000"
OUTPUT_DIR="$DEFAULT_DIR_BASE/7b-conversion-tests/8K/MP1"
MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/configs/7b-context-ext/model_configs/7b_stripedhyena2_base_4M_32k.yml"

CMD="python conversion/interleave_model_states.py \
    --source_dir $SOURCE_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --model_config $MODEL_CONFIG \
    --overwrite"

echo $CMD
eval $CMD
