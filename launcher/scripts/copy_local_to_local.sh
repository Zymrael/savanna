#!/bin/bash

set -euo pipefail

SOURCE=$1
DESTINATION=$2

CMD="rsync -ah --progress $SOURCE $DESTINATION"
echo $CMD
$CMD
