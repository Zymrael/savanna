# zero3: /lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-n16-1M-7b-hybrid-log_evo1-1M-zero3-test/202412210758/
# zero1: /lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-test-n4-v3-hybrid-log_evo1-1M/202412210155/

NODES=(8)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="1M"
ROPE_SCALE="hybrid-log_evo1"

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_v3_nvidia.yml
CONFIG="7b-hybrid-log_evo1-1M"
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/generated/$CONTEXT_LEN/$CONFIG.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=7b-context-extension
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX="${CONTEXT_LEN}-${CONFIG}-test"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$BASE_NAME-$RUN_NAME

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --disable-checkpoint \
    --use-nvte-flash \
    --avoid_record_streams \
    --expandable_segments \
    --enable-heimdall \
    --use-wandb \
    --launcher $LAUNCHER \
    --job-time $JOBTIME \
    --partition $PARTITION \
    --account $ACCOUNT \
    --container $CONTAINER \
    --num-nodes $NUM_NODES \
    --num-gpus $NUM_GPUS \
    --data-config $DATA_CONFIG \
    --model-config $MODEL_CONFIG \
    --train-script $TRAIN_SCRIPT \
    --wandb-project $BASE_NAME \
    --wandb-group $CONTEXT_LEN \
    --wandb-run-name $RUN_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done