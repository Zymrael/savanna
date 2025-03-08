# Training run without checkpointing -- for running NVIDIA dummy jobs

NODES=(1)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext
CONTEXT_LEN="32K"
ROPE_SCALE="log"

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
# DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
#DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_v3_nvidia_test.yml #configs/launcher-test/data_configs/opengenome.yml

#DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_test.yml

# DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/longphase_v3_nvidia.yml
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/data_configs/shortphase_v3_nvidia.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-context-ext/model_configs/tests/evo2_conversion/7b_evo2_mp1_check.yml

#Containers
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=7b-evo2-conversion
LAUNCHER=srun
JOBTIME="01:00:00"
JOBNAME="pretrained-check"

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

OVERRIDES="do_per_ds_valid=False"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$JOBNAME-n$N

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
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
    --wandb-run-name $JOB_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done