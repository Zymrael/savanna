#!/bin/bash

set -euo pipefail

ITERATION=500000
INPUT_CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/global_step$ITERATION" #/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256/40b_train/202410261701/global_step${ITERATION} #"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/global_step$ITERATION"
OUTPUT_CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension/7b_base_4M/universal/global_step${ITERATION}_universal"  ##7b_stripedhyena2_base_4M/global_step${ITERATION}_universal"

echo "Converting from $INPUT_CHECKPOINT_DIR to $OUTPUT_CHECKPOINT_DIR"

# rm -r $OUTPUT_CHECKPOINT_DIR
mkdir -p $OUTPUT_CHECKPOINT_DIR
#pip uninstall deepspeed -y

#DEEPSPEED_DIR="/lustre/fs01/portfolios/dir/users/jeromek/DeepSpeed"
# export PYTHONPATH=$DEEPSPEED_DIR:$PYTHONPATH
# echo $PYTHONPATH
CMD="python tools/ds_to_universal.py \
    --input_folder $INPUT_CHECKPOINT_DIR \
    --output_folder $OUTPUT_CHECKPOINT_DIR \
    --inject_missing_state"
echo $CMD

eval $CMD

# du -sb $INPUT_CHECKPOINT_DIR
# du -sb $OUTPUT_CHECKPOINT_DIR
