#/bin/bash

set -euo pipefail

CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618 #/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-v2/40b_train_v2/202410271619/ #"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256/40b_train/202410261701"
DELIMITER=" ------------------------------------- "
CMD="du -sb $CHECKPOINT_DIR/global_step*"
#echo $CMD
echo "Checkpoint dir $CHECKPOINT_DIR local sizes"
echo $DELIMITER
eval $CMD

echo $DELIMITER

echo "Checkpoint dir $CHECKPOINT_DIR num chkpted files"
for dir in $CHECKPOINT_DIR/global_step*/; do
    NUM_FILES=`ls -l $dir | wc -l`
    echo $dir : $NUM_FILES

done
