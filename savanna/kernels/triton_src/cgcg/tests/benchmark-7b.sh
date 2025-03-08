#!/bin/bash

set -euo pipefail

BS=(8)
FILTER_SIZE=(128)
SEQLEN=8192
D=4096
G=256

BENCH_SCRIPT="refactor-benchmark.py"
for bs in "${BS[@]}"; do
    for filter_size in "${FILTER_SIZE[@]}"; do
        LOG_PATH="7b-bench-BS-${bs}-FILTER_SIZE-${filter_size}-SEQLEN-${SEQLEN}-D-${D}-G-${G}.log"

        echo "Running with bs=$bs and filter_size=$filter_size seqlen=$SEQLEN d=$D g=$G"
        python -u $BENCH_SCRIPT --bs "$bs" --filter_size "$filter_size" --seqlen "$SEQLEN" --d "$D" --g "$G" 2>&1 | tee "$LOG_PATH"
    
    done
done
