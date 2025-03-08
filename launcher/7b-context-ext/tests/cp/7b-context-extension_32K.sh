NODES=(4)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="32K"
ROPE_SCALE="hybrid-log_evo1"

USE_CP=2

if [ $USE_CP -eq 1 ]; then  
    SUFFIX="$ROPE_SCALE-$CONTEXT_LEN-cp2-mp1-mbs1-ac2"
    MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/cp/$CONTEXT_LEN/7b-$ROPE_SCALE-$CONTEXT_LEN-cp2-mp1-mbs1-ac2.yml
elif [ $USE_CP -eq 2 ]; then
    SUFFIX="$ROPE_SCALE-$CONTEXT_LEN-cp2-mp2-mbs2-ac4"
    MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/cp/$CONTEXT_LEN/7b-$ROPE_SCALE-$CONTEXT_LEN-cp2-mp2-mbs2-ac4.yml
else
    SUFFIX="$ROPE_SCALE-$CONTEXT_LEN-cp1-mp2-mbs1-ac4"
    MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/cp/$CONTEXT_LEN/7b-$ROPE_SCALE-$CONTEXT_LEN-cp1-mp2-mbs1-ac4.yml
fi

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_v3_nvidia.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-context-extension-cp-test
LAUNCHER=srun
JOBTIME="02:00:00"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$BASE_NAME-$RUN_NAME

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --disable-checkpoint \
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