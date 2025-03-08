#!/bin/bash

set -euo pipefail

PREFIX_40B="chkpt/40b-train-n256-v2/40b_train_v2"
PREFIX_7B_LOG="chkpt/7b-context-extension-n32-log-32K/7b-log-32K"
PREFIX_7B_LINEAR="chkpt/7b-context-extension-n32-linear-32K/7b-linear-32K"
PREFIX_7B_EVO1="chkpt/7b-context-extension-n32-evo1-32K/7b-evo1-32K"
PREFIX_7B_5X="chkpt/7b-context-extension-n32-5x-32K/7b-5x-32K"
#cat ../../7b-context-extension-n32-linear-32K/202411092309/model_configs/7b-linear-32K.yml | grep location | awk '{print $2}'

PREFIX=$PREFIX_7B_LINEAR
CMD="python scripts/collect_s3_chkpts.py \
    --prefix $PREFIX"

echo $CMD
eval $CMD