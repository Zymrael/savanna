# Training run without checkpointing -- for running NVIDIA dummy jobs

NODES=(1)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="32K"
ROPE_SCALE="log"

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
#DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_v3_nvidia_test.yml #configs/launcher-test/data_configs/opengenome.yml
#DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/shortphase_test.yml
#DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_test.yml
# DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/opengenome.yml

MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/evo2_conversion/7b-mock-mp1-checkpoint.yml
CHECKPOINT_PATH="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-mock-checkpoints/MP2"
#Containers
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-zero1-checkpointing
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX="mp1-mock-checkpoint"

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

# OVERRIDES="save=$CHECKPOINT_PATH"
#--disable-checkpoint \
#   --use-wandb \
     
for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N$SUFFIX

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --overrides $OVERRIDES \
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