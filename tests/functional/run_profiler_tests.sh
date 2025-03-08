#!/bin/bash

echo "Running profiler test"

echo "Testing empty profiler"
python tests/functional/test_profiler.py --model_config tests/test_configs/profiler/empty_profiler_test.yml

echo "Testing torch profiler"
python tests/functional/test_profiler.py --model_config tests/test_configs/profiler/torch_profiler_test.yml

echo "Testing nsys profiler"
python tests/functional/test_profiler.py --model_config tests/test_configs/profiler/nsys_profiler_test.yml