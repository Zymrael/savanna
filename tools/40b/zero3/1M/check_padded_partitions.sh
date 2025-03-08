#!/bin/bash
set -euo pipefail

CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero3/MP64DP16/padded/global_step0"

CMD="python extension-checks/check_padding.py $CHECKPOINT_DIR"

echo $CMD
eval $CMD