#!/bin/bash
set -euo pipefail

BASE_BUCKET="s3://nv-arc-dna-us-east-1"
# TS="202411091748"
# LOCATION=`cat ../../7b-context-extension-n32-log-32K/$TS/model_configs/7b-log-32K.yml | grep location | awk '{print $2}'` #`cat ../../7b-context-extension-n32-linear-32K/202411092309/model_configs/7b-linear-32K.yml | grep location | awk '{print $2}'`
# echo $LOCATION
PREFIX_40B="chkpt/40b-train-n256-8K/40b_train_8K" #"chkpt/40b-train-n256-v2/40b_train_v2"
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

BUCKET=$PREFIX_40B #"chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume" #chkpt/40b-train-n256/40b_train" #chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume
#PREFIX=$PREFIX_40B

BUCKET_NAME=$BASE_BUCKET/$BUCKET
START="${1:-422000}" #12500 + 3125 = 15625
END="${2:-428000}"
BUCKET_NAME=${3:-$BUCKET_NAME}
FLAGS="--recursive --summarize"
S3_CMD="/home/jeromek/aws-cli/bin/aws"
for (( i=START; i<=END; i+=2000 )); do
    STEPS+=("$i")
done


for step in "${STEPS[@]}"; do
    SIZE="$S3_CMD s3 ls $BUCKET_NAME/global_step$step --summarize --recursive | tail -n2"
    echo $SIZE
    eval $SIZE
done
