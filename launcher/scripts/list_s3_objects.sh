#!/bin/bash

BASE_BUCKET="s3://nv-arc-dna-us-east-1"
BUCKET="chkpt/40b-train-n256/40b_train"
BUCKET_NAME=$BASE_BUCKET/$BUCKET
INTERVAL=3600  # Interval in seconds (e.g., 3600s = 1 hour)
LOG_DIR=s3_logs
FLAGS="--recursive --summarize"
mkdir -p $LOG_DIR
S3_CMD="/home/jeromek/aws-cli/bin/aws"
# Infinite loop to periodically check the bucket
while true; do
    # Generate a timestamped log file name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="40b-train-n256_$TIMESTAMP.log"
    
    # List all objects in the bucket and write to the log file
    $S3_CMD s3 ls $BUCKET_NAME --summarize --recursive > "$LOG_DIR/$LOG_FILE"

    # Output confirmation message
    echo "Logged objects from $BUCKET_NAME to $LOG_FILE at $TIMESTAMP"
    break
    # Wait for the specified interval
    sleep $INTERVAL
done
