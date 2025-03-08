#/lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-cp-test-n2-hybrid-log_evo1-8K-mp1-cp2-te-flash-4layer/202412021524/
NODES=(2)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="8K"
ROPE_SCALE="log"
CP_CONFIG="mp1-cp2-te-4layer-p2p-flash"
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/cp/$CONTEXT_LEN/${CP_CONFIG}.yml #mp1-cp2-ring-4layer.yml #mp1-cp2-te-flash-4layer.yml #mp2-cp1-4layer.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-context-extension-cp-test
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX="$ROPE_SCALE-$CONTEXT_LEN-$CP_CONFIG"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$BASE_NAME-$RUN_NAME
 
   # --use-nvte-flash \

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --nvte-debug \
    --disable-checkpoint \
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