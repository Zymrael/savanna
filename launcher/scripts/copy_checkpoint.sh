#!/bin/bash

set -euo pipefail

SOURCE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/"
DEST_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-ext/checkpoint_test/"
mkdir -p $DEST_DIR

CMD="cp -r $SOURCE_DIR $DEST_DIR"

echo $CMD
eval $CMD