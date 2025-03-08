# Example of Heimdall with 7b model
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/7b_shc_post_refactor-mp-dp.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh" 
NUM_NODES=2
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

# Change to your desired launcher: torch, deepspeed, srun
# They are functionally equivalent, srun is the default
# deepspeed has the additional feature of --enable-each-rank-log, which outputs per-rank logs to rank_logs directory
# Best to use `srun` for now, as we will be integrating more performance-oriented features only available with `srun`
LAUNCHER=srun
OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME

# Additional flags that will override user-provided model config
# --use_wandb: wandb monitoring -- make sure to export your WANDB_API_KEY if using wandb, otherwise will get assertion error
# needed to use wandb within container
# --nsys: launches nsys profiling; see `python generated_distributed_launcher.py --help` for all available nsys options 
CMD="python generate_distributed_launcher.py \
$JOB_NAME \
--enable-heimdall \
--output-dir $OUTPUT_DIR \
--launcher $LAUNCHER \
--partition $PARTITION \
--account $ACCOUNT \
--container $CONTAINER \
--num-nodes $NUM_NODES \
--num-gpus $NUM_GPUS \
--data-config $DATA_CONFIG \
--model-config $MODEL_CONFIG \
--train-script $TRAIN_SCRIPT"

echo $CMD
eval $CMD