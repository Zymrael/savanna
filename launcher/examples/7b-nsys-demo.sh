# `nsys` profile of 7b model config
#
# Usage:
# From `launcher` directory
# ./examples/7b-nsys.sh
# This will output: "SLURM script generated: .../$JOB_NAME/.../7b-xxx.sbatch"
# Submit and check logs: `sbatch .../$JOB_NAME/.../7b_xxx.sbatch`
# The nsys reports will be located in output directory `nsys_reports/*rep`

# NOTE: use full path; relative paths won't be mapped correctly in the SLURM script or container
SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/7b_shc_post_refactor-mp-dp.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh" #/lustre/fs01/portfolios/dir/project/dir_arc/containers/clara-discovery+savanna+arc-evo2_efa+nv-latest-cascade-1.5.sqsh"
NUM_NODES=2
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

# NOTE: when using nsys, choose torch or deepspeed as the launcher as these spin up a main process on each node that spawns additional per-gpu processes
# This results in one nsys report per gpu per node. 
# srun, on the other hand, launches `WORLD_SIZE` number of individual processes, resulting in `WORLD_SIZE` number of nsys reports
LAUNCHER=torch
OUTPUT_DIR=$SAVANNA_ROOT/launcher/$JOB_NAME

CMD="python generate_distributed_launcher.py \
$JOB_NAME \
--output-dir $OUTPUT_DIR \
--nsys \
--nsys_warmup_steps 10 \
--nsys_num_steps 1 \
--cuda-memory-usage true \
--trace cuda,nvtx,cudnn,cublas-verbose \
--sample cpu \
--nic-metrics true \
--capture-range cudaProfilerApi \
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