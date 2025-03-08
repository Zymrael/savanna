#!/bin/bash

set -euo pipefail

GLOBAL_STEP=207400 #199400
INPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b_train/model_states/train/global_step${GLOBAL_STEP}"
#INPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b_train/model_states/deepspeed/raw/global_step${GLOBAL_STEP}"
OUTPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b_train/model_states/deepspeed/" #clean/global_step${GLOBAL_STEP}" #deepspeed/updated/global_step${GLOBAL_STEP}"
DS_CHECKPOINT="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b_train/ds_meta_states/global_step${GLOBAL_STEP}/zero_pp_rank_0_mp_rank_00_model_states.pt" #"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b_train/model_states/deepspeed/zero3/global_step${GLOBAL_STEP}/zero_pp_rank_0_mp_rank_00_model_states.pt"
DP_WORLD_SIZE=1
CMD="python update_model_state.py --input_dir $INPUT_DIR --deepspeed_checkpoint $DS_CHECKPOINT --output_dir $OUTPUT_DIR --global_step $GLOBAL_STEP --dp_world_size $DP_WORLD_SIZE"
echo $CMD
$CMD