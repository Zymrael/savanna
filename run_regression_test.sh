#!/bin/bash

# Command-line arguments
DATA_CONFIG="configs/data/opengenome.yml"
CONFIG_1="configs/test/regression_1.yml"
CONFIG_2="configs/test/regression_2.yml"
CONFIG_3="configs/test/regression_3_checkpoint_reload.yml"
LOG_DIR="$(pwd)/logs/regression"
CHECKPOINT_DIR="${LOG_DIR}/checkpoints"
TRAIN_ITERS_1="2000"
TRAIN_ITERS_2="2000"
TRAIN_ITERS_3="50"
CHECKPOINT_RELOAD_TEST=true

TS="$(date +%Y%m%d%H%M)"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run regression tests"
    echo ""
    echo "Options:"
    echo "  --data-config <file>      Path to the data config file (default: $DATA_CONFIG)"
    echo "  --config-1 <file>         Path to the first config file (default: $CONFIG_1)"
    echo "  --config-2 <file>         Path to the second config file (default: $CONFIG_2)"
    echo "  --config-3 <file>         Path to the checkpoint reloading config file (default: $CONFIG_3)"
    echo "  --checkpoint-dir <dir>    Save path to substitute in the first and second config file (default: $CHECKPOINT_DIR)"
    echo "  --train-iters-1 <int>     Number of training iterations to substitute in the first config file"
    echo "  --train-iters-2 <int>     Number of training iterations to substitute in the second config file"
    echo "  --train-iters-3 <int>     Number of training iterations for checkpoint reloading test"
    echo "  --log-dir <dir>           Directory to store log files (default: $LOG_DIR)"
    echo "  --checkpoint-reload-test  Enable checkpoint reloading test"
    echo "  -h, --help                Show this help message and exit"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-config) DATA_CONFIG="$2"; shift ;;
        --config-1) CONFIG_1="$2"; shift ;;
        --config-2) CONFIG_2="$2"; shift ;;
        --config-3) CONFIG_3="$2"; shift ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift ;;
        --train-iters-1) TRAIN_ITERS_1="$2"; shift ;;
        --train-iters-2) TRAIN_ITERS_2="$2"; shift ;;
        --train-iters-3) TRAIN_ITERS_3="$2"; shift ;;
        --log-dir) LOG_DIR="$2"; shift ;;
        --checkpoint-reload-test) CHECKPOINT_RELOAD_TEST=true ;;
         -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

CHECKPOINT_DIR="${CHECKPOINT_DIR}/${TS}"
LOG_DIR="${LOG_DIR}/${TS}"

LOG_DIR_1="${LOG_DIR}/regression_1"
LOG_DIR_2="${LOG_DIR}/regression_2"
LOG_DIR_3="${LOG_DIR}/regression_3_checkpoint_reload"
mkdir -p "$LOG_DIR_1"
mkdir -p "$LOG_DIR_2"
if [ "$CHECKPOINT_RELOAD_TEST" = true ] ; then
    mkdir -p "$LOG_DIR_3"
fi

update_config() {
    local config=$1
    local save_path=$2
    local train_iters=$3
    local log_dir=$4
    local load_path=$5

    UPDATED_CONFIG="${log_dir}/$(basename $config)"
    cp "$config" "$UPDATED_CONFIG"
    sed -i "s|\"save\":.*|\"save\": \"$save_path\",|g" "$UPDATED_CONFIG"
    sed -i "s|\"train-iters\":.*|\"train-iters\": $train_iters,|g" "$UPDATED_CONFIG"
    sed -i "s|\"load\":.*|\"load\": \"$load_path\",|g" "$UPDATED_CONFIG"
    
    echo "Updated config file written to $UPDATED_CONFIG"
    # cat "$UPDATED_CONFIG"
}

# Update both config files
SAVE_PATH_1="${CHECKPOINT_DIR}/regression_1"
SAVE_PATH_2="${CHECKPOINT_DIR}/regression_2"
SAVE_PATH_3="${CHECKPOINT_DIR}/regression_3_checkpoint_reload"
update_config "$CONFIG_1" "$SAVE_PATH_1" "$TRAIN_ITERS_1" "$LOG_DIR_1"
update_config "$CONFIG_2" "$SAVE_PATH_2" "$TRAIN_ITERS_2" "$LOG_DIR_2"
update_config "$CONFIG_3" "$SAVE_PATH_3" "$TRAIN_ITERS_3" "$LOG_DIR_3" "$SAVE_PATH_2"

echo "Logging to $LOG_DIR"

export LOG_DIR_1="${LOG_DIR_1}"
export LOG_DIR_2="${LOG_DIR_2}"
export LOG_DIR_3="${LOG_DIR_3}"
export DATA_CONFIG="${DATA_CONFIG}"
export CONFIG_1="${CONFIG_1}"
export CONFIG_2="${CONFIG_2}"
export CONFIG_3="${CONFIG_3}"
export CHECKPOINT_RELOAD_TEST="${CHECKPOINT_RELOAD_TEST}"

sbatch slurm.regression.sh