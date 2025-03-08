# Generates SLURM batch script that runs pipeline with train script with the 40b model config
#
# Usage:
# From `launcher` directory
# ./examples/40b-demo.sh
# This will output: "SLURM script generated: .../$JOB_NAME/.../40b_test_config.sbatch"
# Submit and check logs: `sbatch .../$JOB_NAME/.../40b_test_config.sbatch`
NODES=(4 8 16 32)
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/40b_base.yml

CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh" #/lustre/fs01/portfolios/dir/project/dir_arc/containers/clara-discovery+savanna+arc-evo2_efa+nv-latest-cascade-1.5.sqsh"
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc
# Change to your desired launcher: torch, deepspeed, srun
# They are functionally equivalent, srun is the default
# deepspeed has the additional feature of --enable-each-rank-log, which outputs per-rank logs to rank_logs directory

BASE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

LAUNCHER=srun
for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N
    OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME
        
    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --use-wandb \
    --output-dir $OUTPUT_DIR \
    --launcher $LAUNCHER \
    --partition $PARTITION \
    --account $ACCOUNT \
    --container $CONTAINER \
    --num-nodes $NUM_NODES \
    --num-gpus $NUM_GPUS \
    --data-config $DATA_CONFIG \
    --model-config $MODEL_CONFIG \
    --train-script $TRAIN_SCRIPT \
    --overrides print_mem_alloc_stats=true \
    --wandb-project $BASE_NAME \
    --wandb-run-name $JOB_NAME"

    echo $CMD
    eval $CMD
    echo -e "\n"

done