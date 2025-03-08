# 40b 512K extension training script
# Tune async_save
 # Check nvte flags
# /lustre/fs01/portfolios/dir/users/jeromek/40b-train-extension-n256-512K_mp8cp16/202412151410

NODES=(256)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/40b/data_configs/longphase_v3_nvidia.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/40b/model_configs/extension/512K/40b_512K_mp8cp16.yml

CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=40b-train-extension
LAUNCHER=srun
JOBTIME="08:00:00"
SUFFIX="-512K_mp8cp16"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N$SUFFIX

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --enable_async_save \
    --use-nvte-flash \
    --enable-heimdall \
    --expandable_segments \
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