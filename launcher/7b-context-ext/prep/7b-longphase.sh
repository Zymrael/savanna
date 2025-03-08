# Training run without checkpointing -- for running NVIDIA dummy jobs

NODES=(4)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="64K"

#ROPE_SCALES=("log" "linear" "evo1" "5x")  # Add all desired ROPE_SCALE values

ROPE_SCALE="evo1"

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_v3_nvidia.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/generated/$CONTEXT_LEN/7b-$ROPE_SCALE-$CONTEXT_LEN.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-context-extension-test
LAUNCHER=srun
JOBTIME="01:00:00"

#TODO: 
# load from checkpoint, 
# fix train loop data loading from checkpoint + finetuning,
# expandable_segments and other memory optimizations
# change dataset to longphase_v3_nvidia.yml
# heimdall
# add special reservation
# enable checkpointing
# remove overrides
# uncomment data config


# --expandable_segments
# --avoid_record_streams
AVOID_RECORD_STREAMS="--avoid_record_streams"
EXPANDABLE="--expandable_segments"

SUFFIX="$ROPE_SCALE-$CONTEXT_LEN-lp${EXPANDABLE}${AVOID_RECORD_STREAMS}"

OVERRIDES="do_per_ds_valid=True eval_per_ds_interval=100 print_mem_alloc_stats=True"


for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$BASE_NAME-$RUN_NAME

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    ${EXPANDABLE} \
    ${AVOID_RECORD_STREAMS} \
    --overrides $OVERRIDES \
    --disable-checkpoint \
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