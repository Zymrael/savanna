#!/bin/bash


# This script runs kernel tests normally
# Any failed tests are subsequently run under triton interpreter (CPU) mode
# Example ./test_runner.sh test_fwd.py
#NOTE: Not triton interpreter not working on triton nightly, see https://github.com/triton-lang/triton/issues/4317

if [ -z "$1" ]; then
  echo "Usage: $0 <test_file>"
  exit 1
fi

test_file=$1
#Remove extension from test_file
test_name="${test_file%.*}"
# Ensure the logs directory exists
logs_dir="logs"
if [ ! -d "$logs_dir" ]; then
  mkdir -p "$logs_dir"
fi

# First test kernels normally
pytest -s --tb=short --cache-clear $test_file 2>&1 | tee logs/${test_name}.log
# Run any failing tests under interpreter mode
TRITON_INTERPRET=1 python run_last_failed.py $test_file 2>&1 | tee logs/${test_name}_interpreter_failed.log
