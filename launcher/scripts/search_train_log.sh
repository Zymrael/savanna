#/bin/bash

set -euo pipefail
# HSD: Rtt/Rnk, Etpt
LOGFILES=`ls /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-v2/202410281731/logs/40b_train_v2/*.log`
PAT=$1
CMD="cat $LOGFILES | grep -n $PAT"
echo $LOGFILES
echo $CMD
eval $CMD