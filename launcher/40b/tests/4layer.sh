# 40b 8K early extension training script
#/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-8K/202411291153/
# /lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-test-n2-4layer_zero3/4layer_zero3/202412160615
#/lustre/fs01/portfolios/dir/users/jeromek/40b-checkpoint-test-n2-4layer_zero3/202412160813/model_configs/4layer_zero3.yml
#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/4layer/zero3/global_step1
#/lustre/fs01/portfolios/dir/users/jeromek/40b-fp8-test-n1-4layer_zero3_mp8/202412171648/
NODES=(2)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml #/configs/40b/data_configs/shortphase_v3_nvidia.yml
CONFIG="4layer_zero3_mp8" # "4layer_zero3"
MODEL_CONFIG=$SAVANNA_ROOT/configs/40b/model_configs/tests/fp8/${CONFIG}.yml

CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=40b-fp8-test
LAUNCHER=srun
JOBTIME="01:00:00"
SUFFIX="${CONFIG}"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$BASE_NAME-$RUN_NAME
#    --enable_async_save \
#  --use-nvte-flash \
    # --nvte-debug \
  
    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --avoid_record_streams \
    --disable-checkpoint \
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
    --wandb-run-name $RUN_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done