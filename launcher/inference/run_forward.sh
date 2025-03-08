
NODES=(2)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/evals/model.py
PARALLEL_CONFIG="MP8CP2"
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/inference/7b-1M-4layer-${PARALLEL_CONFIG}-refactored.yml

#Containers
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=inference
LAUNCHER=torch
JOBTIME="01:00:00"
SUFFIX="4layer-${PARALLEL_CONFIG}"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N-$SUFFIX

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --enable-heimdall \
    --disable-checkpoint \
    --use-nvte-flash \
    --avoid_record_streams \
    --expandable_segments \
    --launcher $LAUNCHER \
    --job-time $JOBTIME \
    --partition $PARTITION \
    --account $ACCOUNT \
    --container $CONTAINER \
    --num-nodes $NUM_NODES \
    --num-gpus $NUM_GPUS \
    --data-config $DATA_CONFIG \
    --model-config $MODEL_CONFIG \
    --train-script $TRAIN_SCRIPT"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done