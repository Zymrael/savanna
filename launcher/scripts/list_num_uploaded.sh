#/bin/bash

set -euo pipefail

LOGFILE=$1
STEP=$2

CMD="grep global_step$STEP $LOGFILE | wc -l"
echo $CMD
eval $CMD