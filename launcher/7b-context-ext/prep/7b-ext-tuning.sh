# Training run without checkpointing -- for running NVIDIA dummy jobs

NODES=(32)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="64K"

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/generated/7b-log-$CONTEXT_LEN.yml

#Containers
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-ext-tuning-$CONTEXT_LEN
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX=""

SEARCH_CONFIG=$SCRIPT_DIR/search_configs/$CONTEXT_LEN.yml

#OVERRIDES="train_micro_batch_size_per_gpu=$MBS checkpoint-num-layers=$AC model_parallel_size=$MP sequence_parallel=$SP"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N$SUFFIX

#    --expandable_segments \

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --search-config $SEARCH_CONFIG \
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
    --wandb-run-name $JOB_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done