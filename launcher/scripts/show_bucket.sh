#!/bin/bash
set -euo pipefail

BUCKET="s3://nv-arc-dna-us-east-1"
# TS="202411091748"
# LOCATION=`cat ../../7b-context-extension-n32-log-32K/$TS/model_configs/7b-log-32K.yml | grep location | awk '{print $2}'` #`cat ../../7b-context-extension-n32-linear-32K/202411092309/model_configs/7b-linear-32K.yml | grep location | awk '{print $2}'`
# echo $LOCATION
PREFIX_40B="chkpt/40b-train-n256-v2/40b_train_v2"
PREFIX_7B_LOG="chkpt/7b-context-extension-n32-log-32K/7b-log-32K"
PREFIX_7B_LINEAR="chkpt/7b-context-extension-n32-linear-32K/7b-linear-32K"
PREFIX_7B_EVO1="chkpt/7b-context-extension-n32-evo1-32K/7b-evo1-32K"
PREFIX_7B_5X="chkpt/7b-context-extension-n32-5x-32K/7b-5x-32K"
PREFIX_64K_EVO="chkpt/7b-context-extension-n32-evo1-64K/7b-evo1-64K"
PREFIX_64K_LINEAR="chkpt/7b-context-extension-n32-linear-64K/7b-linear-64K"
PREFIX_128K_EVO="chkpt/7b-context-extension-n32-evo1-128K/7b-evo1-128K"
PREFIX_128K_LINEAR="chkpt/7b-context-extension-n32-linear-128K/7b-linear-128K"
PREFIX_8K_to_128K_LOG="chkpt/7b-context-extension-n32-v2-8K_to_128K_log/7b-log-128K"
PREFIX_V2_EVO1="chkpt/7b-context-extension-n32-v2-evo1-32K/7b-evo1-32K"
TRAIN_40b="chkpt/40b-train-n256-v2/40b_train_v2"
#BUCKET=$PREFIX_7B_LINEAR #"chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume" #chkpt/40b-train-n256/40b_train" #chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume
PREFIX=$PREFIX_8K_to_128K_LOG

BUCKET_NAME=$BUCKET/$PREFIX
FLAGS="--recursive --summarize"
S3_CMD="/home/jeromek/aws-cli/bin/aws"

STEP=12500
# Get timestamp to the minute
TIMESTAMP=$(date +"%Y%m%d_%H%M")
# Get the basename of the prefix
BASENAME=$(basename $PREFIX)
# Get the dirname of the prefix
DIRNAME=$(dirname $PREFIX)
LOG_DIR=s3_logs
LS_CMD="$S3_CMD s3 ls --recursive --summarize $BUCKET_NAME"
# echo $DIRNAME
echo $LS_CMD

OUTPUT_DIR=$LOG_DIR/$DIRNAME

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

eval $LS_CMD > ${OUTPUT_DIR}/${TIMESTAMP}.log

cat ${OUTPUT_DIR}/${TIMESTAMP}.log | awk '{print $4}' | xargs -d '\n' -n 1 dirname | sort -hr | uniq
# | cut -c32- | xargs -d '\n' -n 1 dirname | sort -hr | uniq
#aws s3 ls s3://<your bucket>/<path>/<to>/ | awk '{print $4}' | xargs' -d '\n' -n 1 dirname | sort -hr | uniq
# LS_STEP_CMD="$S3_CMD s3 ls $BUCKET_NAME --recursive --summarize"
# echo $LS_STEP_CMD
# eval $LS_STEP_CMD

# SIZE_CMD="$LS_STEP_CMD | tail -n2"
# echo $SIZE_CMD
# eval $SIZE_CMD
