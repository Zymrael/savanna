#!/bin/bash

set -euo pipefail

ENTITY="hyena"
PROJECT="40b-train"
RUN_PATTERN="40b-train-n256-v2"
MOST_RECENT=20
OUTPUT_DIR="wandb_runs"

CMD="python scripts/export_wandb_runs.py \
    --entity $ENTITY \
    --project $PROJECT \
    --run $RUN_PATTERN \
    --most_recent $MOST_RECENT \
    --output_dir $OUTPUT_DIR"
    
echo $CMD
eval $CMD