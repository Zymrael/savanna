# 40b 512K extension training script
# Tune async_save
 # Check nvte flags
# /lustre/fs01/portfolios/dir/users/jeromek/40b-train-extension-n256-512K/202412131556/
#
#/lustre/fs01/portfolios/dir/users/jeromek/40b-train-extension-n256-32K-mp8cp8/202412141334/
#/lustre/fs01/portfolios/dir/users/jeromek/40b-train-extension-n256-40b_32K_mp8cp8-no-hyena/202412141736

NODES=(64)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/40b/data_configs/longphase_v3_nvidia.yml
CONTEXT_LEN="32layer"
CONFIG="40b_${CONTEXT_LEN}_extended_mp8cp2-no-load"
MODEL_CONFIG=$SAVANNA_ROOT/configs/40b/model_configs/tests/${CONTEXT_LEN}/${CONFIG}.yml

CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=40b-train-extension
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX="${CONFIG}-test"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N-$SUFFIX

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --use-nvte-flash \
    --enable-heimdall \
    --enable_async_save \
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