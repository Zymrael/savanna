#!/bin/bash

# Get the total number of lines in the input file
total=$(wc -l < $1)
# Calculate 40% of the total
sample=$((total * 40 / 100))

# Get the base name of the input file without extension
base_name=${1%.jsonl}

# Use shuf to get the sample lines and write them to the output file
shuf -n $sample $1 > "$base_name.sampled.jsonl"

# to run
# for file in $(ls /data/github/filtered*); do bash sample_github.sh "$file" & done