#!/bin/bash

# Usage: bash tools/evaluate_per_ds.sh

# Input variables.
checkpoint_dir=/scratch/hielab/brianhie/checkpoint/evo2/7b_13h_8m_8s_3a_cascade15
model_config_file=configs/model/evo2/7b_13h_8m_8s_3a_cascade15.yml
data_config_file=configs/data/val_version_opengenome2.yml

# Get all checkpoint iterations in the checkpoint directory.
readarray -t iterations < <(ls $checkpoint_dir | \
                            grep -v universal | \
                            sed -n 's/^global_step\([0-9]*\)$/\1/p' | \
                            sort -n)

rm -rf evaluate_per_ds*.log

# Actually compute the values and log them.
for iter in "${iterations[@]}"
do
    echo "Computing values for iteration: $iter"
    sed -i "s/\"iteration\": [0-9]*,/\"iteration\": $iter,/" $model_config_file
    python ./launch.py \
           tools/evaluate_per_ds.py \
           -d configs \
           ${data_config_file#configs/} \
           ${model_config_file#configs/} \
           >> evaluate_per_ds.log 2>&1
done

# Extract the values from the log, and plot.
grep -E "Evaluating |text_CharLevelTokenizer_document|results at the end of training for val data" evaluate_per_ds.log | \
    grep -v config_files | \
    grep -v data_paths \
         > evaluate_per_ds_filtered.log

python tools/evaluate_per_ds_plot_results.py evaluate_per_ds_filtered.log
