# 40b 8K early extension training script
#/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-8K/202411291153/
 
NODES=(2)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml #/configs/40b/data_configs/shortphase_v3_nvidia.yml
CONFIG="4layer_mp8-cp1-zero3" #"4layer-mp8-cp1-zero3", "4layer-mp4-cp2-zero1-ring", "4layer-mp4-cp2-zero3-ring", "4layer-mp4-cp2-zero1-te-flash", "4layer-mp4-cp2-zero1-ring" 
MODEL_CONFIG=$SAVANNA_ROOT/configs/40b/model_configs/tests/z3/${CONFIG}.yml

CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=40b-extension-test
LAUNCHER=srun
JOBTIME="06:00:00"
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
    --use-nvte-flash \
    --nvte-debug \
    --disable-checkpoint \
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