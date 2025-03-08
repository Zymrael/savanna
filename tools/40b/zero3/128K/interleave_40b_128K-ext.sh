#!/bin/bash

set -euo pipefail

SOURCE_MP_SIZE=8
SOURCE_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/zero1/MP8/global_step12500
OUTPUT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero1/MP${SOURCE_MP_SIZE}
MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-long-ext/configs/40b/model_configs/extension/128K/40b_128K_no_rc.yml

CMD="python conversion/interleave_model_states.py --source_dir $SOURCE_DIR --output_dir $OUTPUT_DIR --model_config $MODEL_CONFIG --mp_size $SOURCE_MP_SIZE"

echo $CMD
eval $CMD

#Outputs: /lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero1/global_step12500