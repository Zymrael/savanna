#!/bin/bash

# Check if the log file name is provided as an argument
if [ $# -eq 0 ]; then
  echo "Please provide the log file name as an argument"
  exit 1
fi

# Assign the log file name to a variable
log_file=$1

# Loop through each line in the log file
while read line; do
  # Check if the line contains iteration and lm_loss but not lm_loss_ppl
  if [[ $line == *"iteration"* && $line == *"lm_loss"* && $line != *"lm_loss_ppl"* ]]; then
    # Extract the iteration and lm_loss values using awk
    iteration=$(echo $line | awk -F 'iteration' '{print $2}' | awk -F '/' '{print $1}' | tr -d ' ')
    lm_loss=$(echo $line | awk -F 'lm_loss:' '{print $2}' | awk -F '|' '{print $1}' | tr -d ' ')
    # Print the iteration and lm_loss values
    echo "$iteration,$lm_loss"
  fi
done < "$log_file"