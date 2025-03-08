#!/bin/bash
set -euo pipefail

SCRIPT1="/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-v2/202410281641/scripts/40b_train_v2.sbatch"
SCRIPT2="/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-v2/202410271619/scripts/40b_train_v2.sbatch"

diff -w --suppress-common-lines $SCRIPT1 $SCRIPT2