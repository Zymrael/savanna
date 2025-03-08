# `nsys` profile of 40b model config
#
# Usage:
# From `launcher` directory
# ./examples/40b-nsys-demo.sh
# This will output: "SLURM script generated: .../$JOB_NAME/.../40b-xxx.sbatch"
# Submit and check logs: `sbatch .../$JOB_NAME/.../40b_xxx.sbatch`
# The nsys reports will be located in output directory `nsys_reports/*rep`

SAVANNA_ROOT=$(realpath ..)
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/launcher-test/data_configs/opengenome.yml
MODEL_CONFIG=$SAVANNA_ROOT/configs/launcher-test/model_configs/40b_test_config.yml
JOB_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
CONTAINER="/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh"
NUM_NODES=4
NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

# NOTE: when using nsys, choose torch or deepspeed as the launcher as these spin up a main process on each node that spawns additional per-gpu processes
# This results in one nsys report per gpu per node. 
# srun, on the other hand, launches `WORLD_SIZE` number of individual processes, resulting in `WORLD_SIZE` number of nsys reports
LAUNCHER=torch

# 40b profile reports are very large, so we use minimal set of profiling features
# Note that we're using the default output-dir in this case, which is /lustre/fs01/portfolios/dir/users/$USER
CMD="python generate_distributed_launcher.py \
$JOB_NAME \
--nsys \
--nsys_warmup_steps 10 \
--nsys_num_steps 1 \
--cuda-memory-usage false \
--trace cuda,nvtx \
--sample none \
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