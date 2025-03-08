# 40b 512K extension training script

NODES=(4)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
CONTEXT_LEN="8K"
CONFIG="40b_${CONTEXT_LEN}_extended_mp16cp1_repartitioned_z3_interleaved_padded"
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
    --disable-checkpoint \
    --avoid_record_streams \
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