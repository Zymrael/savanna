#!/bin/bash
set -euo pipefail

# GLOBAL VARIABLES

BUCKET="s3://nv-arc-dna-us-east-1"
PREFIX_8K_to_128K_LOG="chkpt/7b-context-extension-n32-v2-8K_to_128K_log/7b-log-128K"
CONTEXT_LENS=("32K" "64K" "128K")
ROPE_SCALES=("evo1" "hybrid-log_evo1")
FLAGS="--recursive --summarize"
S3_CMD="/home/jeromek/aws-cli/bin/aws"
STEP=12500
LOG_DIR=s3_logs

list_s3_bucket() {

    local PREFIX=$1
    local BUCKET_NAME=$BUCKET/$PREFIX

    local TIMESTAMP=$(date +"%Y%m%d_%H")
    local LS_CMD="$S3_CMD s3 ls $FLAGS $BUCKET_NAME"
    local OUTPUT_DIR=$LOG_DIR/$(dirname $PREFIX)
    if [ ! -d $OUTPUT_DIR ]; then
        mkdir -p $OUTPUT_DIR
    fi

    echo $LS_CMD
    eval $LS_CMD > ${OUTPUT_DIR}/${TIMESTAMP}.log

    # cat ${OUTPUT_DIR}/${TIMESTAMP}.log | awk '{print $4}' | xargs -d '\n' -n 1 dirname | sort -hr | uniq
    SUMMARIZE="cat ${OUTPUT_DIR}/${TIMESTAMP}.log | tail -n2"
    #echo $SUMMARIZE
    eval $SUMMARIZE

   echo -e ""
}

for CONTEXT_LEN in "${CONTEXT_LENS[@]}"; do
    for ROPE_SCALE in "${ROPE_SCALES[@]}"; do
        PREFIX="chkpt/7b-context-extension-n32-v2-$ROPE_SCALE-$CONTEXT_LEN/7b-$ROPE_SCALE-$CONTEXT_LEN"
        # BUCKET_NAME=$BUCKET/$PREFIX
        # Bold the bucket name
        echo -e "\033[1m$BUCKET/$PREFIX\033[0m"
        list_s3_bucket $PREFIX
        if [ $? -ne 0 ]; then
            echo "Error processing $BUCKET_NAME"
        fi
    done
done

# Write a function for the following
        # BUCKET_NAME=$BUCKET/$PREFIX
        

        # # Get timestamp to the minute
        # TIMESTAMP=$(date +"%Y%m%d_%H")
        # # Get the basename of the prefix
        # BASENAME=$(basename $PREFIX)
        # # Get the dirname of the prefix
        # DIRNAME=$(dirname $PREFIX)
        # LS_CMD="$S3_CMD s3 ls --recursive --summarize $BUCKET_NAME"
        # # echo $DIRNAME
        # echo $LS_CMD

        # OUTPUT_DIR=$LOG_DIR/$DIRNAME

        # if [ ! -d $OUTPUT_DIR ]; then
        #     mkdir -p $OUTPUT_DIR
        # fi

        # eval $LS_CMD > ${OUTPUT_DIR}/${TIMESTAMP}.log

        # cat ${OUTPUT_DIR}/${TIMESTAMP}.log | awk '{print $4}' | xargs -d '\n' -n 1 dirname | sort -hr | uniq
        # cat ${OUTPUT_DIR}/${TIMESTAMP}.log | tail -n2 

# PREFIX_V2_EVO1="chkpt/7b-context-extension-n32-v2-evo1-32K/7b-evo1-32K"
# PREFIX_V2_HYBRID_EVO1="chkpt/7b-context-extension-n32-v2-evo1-32K/7b-evo1-32K"

# #BUCKET=$PREFIX_7B_LINEAR #"chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume" #chkpt/40b-train-n256/40b_train" #chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume
# PREFIX=$PREFIX_8K_to_128K_LOG

# BUCKET_NAME=$BUCKET/$PREFIX
# FLAGS="--recursive --summarize"
# S3_CMD="/home/jeromek/aws-cli/bin/aws"

# STEP=12500
# # Get timestamp to the minute
# TIMESTAMP=$(date +"%Y%m%d_%H%M")
# # Get the basename of the prefix
# BASENAME=$(basename $PREFIX)
# # Get the dirname of the prefix
# DIRNAME=$(dirname $PREFIX)
# LOG_DIR=s3_logs
# LS_CMD="$S3_CMD s3 ls --recursive --summarize $BUCKET_NAME"
# # echo $DIRNAME
# echo $LS_CMD

# OUTPUT_DIR=$LOG_DIR/$DIRNAME

# if [ ! -d $OUTPUT_DIR ]; then
#     mkdir -p $OUTPUT_DIR
# fi

# eval $LS_CMD > ${OUTPUT_DIR}/${TIMESTAMP}.log

# cat ${OUTPUT_DIR}/${TIMESTAMP}.log | awk '{print $4}' | xargs -d '\n' -n 1 dirname | sort -hr | uniq
# # | cut -c32- | xargs -d '\n' -n 1 dirname | sort -hr | uniq
# #aws s3 ls s3://<your bucket>/<path>/<to>/ | awk '{print $4}' | xargs' -d '\n' -n 1 dirname | sort -hr | uniq
# # LS_STEP_CMD="$S3_CMD s3 ls $BUCKET_NAME --recursive --summarize"
# # echo $LS_STEP_CMD
# # eval $LS_STEP_CMD

# # SIZE_CMD="$LS_STEP_CMD | tail -n2"
# # echo $SIZE_CMD
# # eval $SIZE_CMD
