#!/bin/bash
set -euo pipefail

BASE_BUCKET="s3://nv-arc-dna-us-east-1"
PREFIX_40B="chkpt/40b-train-n256-v2/40b_train_v2"
# PREFIX_7B_LOG="chkpt/7b-context-extension-n32-log-32K/7b-log-32K"
# PREFIX_7B_LINEAR="chkpt/7b-context-extension-n32-linear-32K/7b-linear-32K"
# PREFIX_7B_EVO1="chkpt/7b-context-extension-n32-evo1-32K/7b-evo1-32K"
# PREFIX_7B_5X="chkpt/7b-context-extension-n32-5x-32K/7b-5x-32K"
# PREFIX_64K_EVO="chkpt/7b-context-extension-n32-evo1-64K/7b-evo1-64K"
# PREFIX_64K_LINEAR="chkpt/7b-context-extension-n32-linear-64K/7b-linear-64K"
# PREFIX_128K_EVO="chkpt/7b-context-extension-n32-evo1-128K/7b-evo1-128K"
# PREFIX_128K_LINEAR="chkpt/7b-context-extension-n32-linear-128K/7b-linear-128K"

BUCKET=$PREFIX_40B #"chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume" #chkpt/40b-train-n256/40b_train" #chkpt/7b-ablations-n32/7b_stripedhyena2_base_4M_resume
STEP=426000
TAG=global_step$STEP
#for BUCKET in $PREFIX_64K_EVO $PREFIX_64K_LINEAR $PREFIX_128K_EVO $PREFIX_128K_LINEAR; do
BUCKET_NAME="$BASE_BUCKET/$BUCKET"
FLAGS="--recursive --summarize"
S3_CMD="/home/jeromek/aws-cli/bin/aws"
STATS="$S3_CMD s3 ls $BUCKET_NAME/$TAG --summarize --recursive | tail -n2"
echo $BUCKET_NAME/$TAG
echo $STATS
eval $STATS

