# Training run without checkpointing -- for running NVIDIA dummy jobs
# /lustre/fs01/portfolios/dir/users/jeromek/7b-context-extension-n32-v2-hybrid-log_evo1-32K/202411141808

NODES=(4)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="32K"
ROPE_SCALE="evo1" #ROPE_SCALES=("evo1" "hybrid-linear_evo1", "hybrid-log_evo1", "hybrid_5x_evo1") 

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/sp/7b-$ROPE_SCALE-$CONTEXT_LEN.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-context-extension-sp-test
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX="$ROPE_SCALE-$CONTEXT_LEN"

#TODO: 
# expandable_segments and other memory optimizations
# change dataset to longphase_v3_nvidia.yml
# add special reservation
# FIX checkpoint store in generated config
OVERRIDES="sequence_parallel=True permute_glu=False"
RUN_CFG_STR=`tr ' ' '__' <<< $OVERRIDES`
RUN_CFG_STR=`tr '=' '-' <<< $RUN_CFG_STR`

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$RUN_CFG_STR-$SUFFIX
    JOB_NAME=$BASE_NAME-$RUN_NAME

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --overrides $OVERRIDES \
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