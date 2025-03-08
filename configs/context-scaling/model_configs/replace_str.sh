#!/bin/bash

set -euo pipefail

DIR=7b/updated
find ${DIR} -iname "*.yml" -exec sed -i 's/flash_v2/flash_te/g' {} \;