# Final 40b training script
# 
NODES=(256)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/40b/shortphase_v3_nvidia.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/40b/40b_train_8K.yml

CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=40b-train
LAUNCHER=srun
JOBTIME="2-00:00:00"
SUFFIX="-8K"

CHECKPOINT_PATH="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-v2/40b_train_v2/202410271619"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N$SUFFIX

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --enable-heimdall \
    --enable_async_save \
    --heimdall_log_straggler \
    --expandable_segments \
    --checkpoint_path $CHECKPOINT_PATH \
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