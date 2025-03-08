#!/bin/bash
set -euo pipefail

BASE_BUCKET="s3://nv-arc-dna-us-east-1"
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
PREFIX_V3_hybrid_32K="chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-32K/7b-hybrid-log_evo1-32K"
PREFIX_V3_hybrid_64K="chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-64K/7b-hybrid-log_evo1-64K" #chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-64K/7b-hybrid-log_evo1-64K
PREFIX_V3_hybrid_128K="chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-128K/7b-hybrid-log_evo1-128K" #chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-128K/7b-hybrid-log_evo1-128K
PREFIX_V3_hybrid_256K="chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-256K/7b-hybrid-log_evo1-256K" #chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-256K/7b-hybrid-log_evo1-256K
PREFIX_V3_hybrid_512K="chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-512K-cp-fix/7b-hybrid-log_evo1-512K-cp-fix" #chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-512K/7b-hybrid-log_evo1-512K
PREFIX_40B_128K="chkpt/40b-train-extension-n256-128K_no_recycle_avoid_streams/40b_128K_no_rc"

for BUCKET in $PREFIX_40B_128K; do
    BUCKET_NAME="$BASE_BUCKET/$BUCKET"
    FLAGS="--recursive --summarize"
    S3_CMD="/home/jeromek/aws-cli/bin/aws"
    STATS="$S3_CMD s3 ls $BUCKET_NAME/global_step12500 --summarize --recursive | tail -n2"
    echo $BUCKET/global_step12500
    echo $STATS
    eval $STATS
done
#s3://nv-arc-dna-us-east-1/chkpt/7b-context-extension-n32-v3-hybrid-log_evo1-512K/7b-hybrid-log_evo1-512K/global_step12500