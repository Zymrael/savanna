import os
import random
from pathlib import Path

from launcher.constants import (
    EVO_ONLY_CONTAINER,
    EXPANDABLE_SEGMENTS,
    INSTALL_CP_REQS,
    MAKE_DATA_HELPERS,
    MAKE_TRITON_CACHE_DIR,
    RECORD_STREAMS_ENV_VAR,
    UPDATE_PYTHONPATH,
)


def _generate_slurm_header(model_config, args, CMDS, num_configs=1):
    log_dir = args.output_dir / args.log_dir / Path(model_config).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    OUTPUT = (
        f"--output={log_dir}/slurm-%a-%N-%J.out" if args.job_array else f"--output={log_dir}/slurm-%N-%J.out"
    )
    ERROR = (
        f"--error={log_dir}/slurm-%a-%N-%J.err" if args.job_array else f"--error={log_dir}/slurm-%N-%J.err"
    )
    tasks_per_node = args.num_gpus if args.launcher == "srun" else 1
    SLURM_HEADER = f"""#!/bin/bash

#SBATCH --job-name={args.job_name}
#SBATCH --partition={args.partition}
#SBATCH --nodes={args.num_nodes}
#SBATCH --gres=gpu:{args.num_gpus}
#SBATCH --time={args.job_time}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --mem={args.mem}
#SBATCH {OUTPUT}
#SBATCH {ERROR}
"""
    if args.cpus_per_task is not None:
        SLURM_HEADER += f"#SBATCH --cpus-per-task={args.cpus_per_task}\n"
    if args.account is not None:
        SLURM_HEADER += f"#SBATCH --account={args.account}\n"
    if args.job_array:
        SLURM_HEADER += f"#SBATCH --array=0-{num_configs - 1}\n"
    if args.exclusive:
        SLURM_HEADER += "#SBATCH --exclusive\n"
        
    if args.enable_heimdall:
        SLURM_HEADER += "#SBATCH --dependency=singleton\n"
        SLURM_HEADER += "#SBATCH --comment='{\"APS\": {}}'\n"
    # Fail fast
    SLURM_HEADER += "\nset -eo pipefail\n"

    CMDS += [SLURM_HEADER]
    return SLURM_HEADER, CMDS


def _generate_slurm_info(model_config: Path, args, CMDS):
    if args.master_port is None:
        master_port = random.randint(10000, 55000)
    else:
        master_port = args.master_port
    MASTER_PORT = f"""MASTER_PORT=$(({master_port} + ${'{SLURM_ARRAY_TASK_ID:-0}'}))"""
    if args.job_array:
        hostlist = f'"{args.output_dir}/{args.hostlist}-$SLURM_ARRAY_TASK_ID"'
    else:
        hostlist = f'"{args.output_dir}/{model_config.stem}-{args.hostlist}"'

    SLURM_INFO = f"""
# Environment info, for debugging
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES
NTASKS=$SLURM_NTASKS
NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
echo "NNODES: $NNODES NGPUS: $GPUS_PER_NODE NTASKS: $NTASKS NTASKS_PER_NODE: $NTASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

HOSTLIST={hostlist}
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo $NODELIST | tr ' ' '\\n' > $HOSTLIST
cat $HOSTLIST

# These are distributed env vars expected by `savanna`
# LOCAL_RANK, RANK, and WORLD_SIZE will be set by the launcher, either `torchrun` or `deepspeed.launcher.launch`
# In the case of `srun`, we need to set these explicitly, of which LOCAL_RANK and RANK are only known during `srun` execution
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
LOCAL_WORLD_SIZE=$GPUS_PER_NODE
export WORLD_SIZE
export LOCAL_WORLD_SIZE
# Note that the container defines its own WORLD_SIZE env var which container-env doesn't seem to be able to override
# We set GLOBAL_NUM_GPUS as a custom env var which the container can pick up
GLOBAL_NUM_GPUS=$WORLD_SIZE
export GLOBAL_NUM_GPUS
echo "WORLD_SIZE: $WORLD_SIZE LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"
MASTER_NODE=$(head -n 1 $HOSTLIST)
export MASTER_NODE
MASTER_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
export MASTER_NODE_IP
{MASTER_PORT}
export MASTER_PORT
echo "MASTER_NODE: $MASTER_NODE, MASTER_NODE_IP: $MASTER_NODE_IP, MASTER_PORT: $MASTER_PORT"

"""

    CMDS += [SLURM_INFO]
    return SLURM_INFO, CMDS


def _generate_env_setup(model_config, args, CMDS):
   
    output_dir = args.output_dir
    ENV_SETUP = f"""
# for wandb logging
export WANDB_API_KEY
ROOT_DIR="{args.root.absolute().as_posix()}"
cd $ROOT_DIR

CONTAINER="{args.container}"
OUTPUT_DIR="{output_dir}"

CHECKPOINT_DIR="{args.checkpoint_dir}"
LOG_DIR="$OUTPUT_DIR/{args.log_dir}/{Path(model_config).stem}"

UPDATE_PYTHONPATH="{UPDATE_PYTHONPATH}"
export UPDATE_PYTHONPATH
"""
    if args.enable_heimdall:
         # log to ${SLURM_LOG_DIR}/aps.log
        SRUN_ARGS = "-l -u --output $LOG_DIR/$APS_LOG_FILE"
    else:
        if args.job_array:
            OUTPUT = "--output $LOG_DIR/srun-%a-%N-%J.out"
            ERROR = "--error $LOG_DIR/srun-%a-%N-%J.err"
        else:
            OUTPUT = "--output $LOG_DIR/srun-%N-%J.out"
            ERROR = "--error $LOG_DIR/srun-%N-%J.err"
        SRUN_ARGS = f"{OUTPUT} \\\n{ERROR}"

    if args.pyxis:
        if args.data_dir is not None:
            CMDS += [f'export DATA_DIR="{args.data_dir}"\n']
        assert args.container is not None
        SRUN_ARGS += " \\\n--container-image $CONTAINER"
        if args.container_workdir is not None:
            SRUN_ARGS += f" \\\n--container-workdir {args.container_workdir}"
        if args.container_mounts is not None:
            SRUN_ARGS += f" \\\n--container-mounts {args.container_mounts}"
        
        #defaults to true
        should_not_mount_home = not args.mount_home
        if should_not_mount_home:
            SRUN_ARGS += " \\\n--no-container-mount-home"
    # if args.container_env is not None:
    #     SRUN_ARGS += f" --container-env {args.container_env}"
    ENV_SETUP += f'SRUN_ARGS="{SRUN_ARGS}"\n'

    CMDS += [ENV_SETUP]
    return ENV_SETUP, CMDS


def _generate_parser_cmd(model_config: Path, data_config: Path, args, CMDS):
    run_id = model_config.stem if args.search_config is not None else args.wandb_run_name or model_config.stem
    
    if args.job_array:
        MODEL_CONFIG_CMD = (
            'MODEL_CONFIG=$(ls $MODEL_CONFIG_DIR/*.yml | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")'
        )
        RUN_ID = 'file_base=$(basename $MODEL_CONFIG)\nRUN_ID="${file_base%.*}"'
        TRAIN_ARGS_OUTPUT = f'''TRAIN_ARGS_OUTPUT_PATH="$OUTPUT_DIR/{Path(args.train_args_output).stem}-$SLURM_ARRAY_TASK_ID.txt"'''
    else:
        MODEL_CONFIG_CMD = f"""MODEL_CONFIG="$MODEL_CONFIG_DIR/{os.path.basename(model_config)}"
        """.strip()
        RUN_ID = f'RUN_ID="{run_id}"'
        TRAIN_ARGS_OUTPUT = f'''TRAIN_ARGS_OUTPUT_PATH="$OUTPUT_DIR/{run_id}-{os.path.basename(args.train_args_output)}"'''

    PARSER_ENV = f"""
# 1. Parse configs and generate train args string to pass to launcher
# Only executed on master node
# - parses and writes formatted args to $TRAIN_ARGS_OUTPUT_PATH
# - compiles data helpers for data loading
# - since all downstream training processes access same savanna root, 
# this should only be done once to prevent race conditions.
# - checks that global_num_gpus, which savanna depends on to set up parallelisms
# is correctly set based on env vars.
# We explicitly set WORLD_SIZE above and implement additional checks
# - In the case of torchrun / deepspeed, `$SLURM_NTASKS * $SLURM_JOB_NUM_NODES == $SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE`
# - In the case of `srun`, `$SLURM_NTASKS == $SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE`
DATA_CONFIG_DIR="{args.data_config_dir}"
MODEL_CONFIG_DIR="{args.model_configs_dir}"
DATA_CONFIG="$DATA_CONFIG_DIR/{os.path.basename(data_config)}"
{MODEL_CONFIG_CMD}
{RUN_ID}
{TRAIN_ARGS_OUTPUT}
TRAIN_SCRIPT="{args.train_script}"
export DATA_CONFIG
export MODEL_CONFIG
export TRAIN_ARGS_OUTPUT_PATH
export TRAIN_SCRIPT
"""
    CMDS += [PARSER_ENV]
    should_add_conda = args.pyxis and args.container == EVO_ONLY_CONTAINER
    if should_add_conda:
        CONDA_CMD = 'CONDA_CMD="conda init && source ~/.bashrc && conda activate evo2"\n'
        CMDS += [CONDA_CMD]

    if args.use_wandb:
        WANDB_ARGS = _make_wandb_args(args)
    else:
        WANDB_ARGS = ""

    PARSER = f"""
PARSER_CMD="python {args.config_parser} \\
{args.train_script} \\
$DATA_CONFIG \\
$MODEL_CONFIG \\
--hostlist $HOSTLIST \\
--train-args-output $TRAIN_ARGS_OUTPUT_PATH{WANDB_ARGS}"
"""

    ADDITIONAL_SRUN_ARGS = "--nodes 1 --ntasks 1 -w $MASTER_NODE"
    BASE_CMD = f"{UPDATE_PYTHONPATH} && {MAKE_DATA_HELPERS} && ${{PARSER_CMD}}"

    if args.use_next:
        BASE_CMD = f"pip install --editable nvidia-resiliency-ext && {BASE_CMD}"

    PARSER += f"""PARSER_CMD="srun $SRUN_ARGS {ADDITIONAL_SRUN_ARGS} bash -c '{BASE_CMD}'"
"""

    PARSER += """
echo $PARSER_CMD
eval $PARSER_CMD
"""

    CMDS += [PARSER]
    return PARSER, CMDS

def _generate_heimdall_cmd(model_config, args, CMDS):
    # heimdall: Create the eventual srun output for the otel collector, and symlink back to launcher.
    log_dir = args.output_dir / args.log_dir / Path(model_config).stem

    log_dir = args.output_dir / args.log_dir / Path(model_config).stem

    if args.enable_heimdall:
        HEIMDALL_CMD = f"""
# heimdall: Create the eventual srun output for the otel collector, and symlink back to launcher.
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
export APS_LOG_DIR=/home/svc-aps-wfc/otelfilelog_paths/dev
export APS_LOG_FILE=${{SLURM_JOB_NAME}}_${{SLURM_JOB_ID}}_${{DATETIME}}.log
touch ${{APS_LOG_DIR}}/${{APS_LOG_FILE}}
ln -sf {log_dir}/${{APS_LOG_FILE}} ${{APS_LOG_DIR}}/${{APS_LOG_FILE}} 
ln -sf {log_dir}/${{APS_LOG_FILE}} ${{APS_LOG_DIR}}/${{APS_LOG_FILE}} 
"""
        CMDS += HEIMDALL_CMD
    return


def _make_nsys_cmd(args):
    nsys_report_path = f"{args.output_dir}/nsys_reports"
    print(f"nsys reports will be written to {nsys_report_path}")
    if not os.path.exists(nsys_report_path):
        os.makedirs(nsys_report_path, exist_ok=True)

    NSYS_CMD = f"""nsys profile \\
--gpu-metrics-device={args.gpu_metrics_device} \\
--cuda-memory-usage={args.cuda_memory_usage} \\
--cudabacktrace={args.cudabacktrace} \\
--python-sampling={args.python_sampling} \\
--capture-range={args.capture_range} \\
--stats={args.stats} \\
--nic-metrics={args.nic_metrics} \\
--show-output={args.show_output} \\
--trace={args.trace} \\
--sample={args.sample} \\
--output={nsys_report_path}/{args.job_name}-%p \\
--force-overwrite={args.force_overwrite} \\
--stop-on-exit={args.stop_on_exit} \\
--wait={args.wait} \\
"""
    return NSYS_CMD


def _make_wandb_args(args):
    # Note the space at the front of wandb_project, this is intentional
    return " ".join(
        [
            " ",
            "--wandb_project",
            args.wandb_project,
            "--wandb_group",
            args.wandb_group,
            "--wandb_run_name",
            "$RUN_ID",
        ]
    )


def _make_torch_launcher(args):
    RDZV_ID = random.randint(0, 10000)
    if args.nsys:
        NSYS_CMD = _make_nsys_cmd(args)
    else:
        NSYS_CMD = ""
    LAUNCHER = f"""
# 2. Launcher
# Runs `torchrun` once on each node
# `torchrun` sets the distributed env (RANK, WORLD_SIZE, etc.) and spawns processes for each local GPU
NSYS_CMD="{NSYS_CMD}"
PROGRAM_CMD="torchrun \\
--nproc_per_node $GPUS_PER_NODE \\
--nnodes $NNODES \\
--master_addr $MASTER_NODE \\
--master_port $MASTER_PORT \\
--node_rank \\$SLURM_PROCID \\
--rdzv_id {RDZV_ID} \\
--rdzv_endpoint $MASTER_NODE:$MASTER_PORT \\
--rdzv_backend {args.rdzv_backend} \\
--max-restarts {args.max_restarts} \\
$TRAIN_SCRIPT \\$(<$TRAIN_ARGS_OUTPUT_PATH)"
export NSYS_CMD
export PROGRAM_CMD
LAUNCHER_CMD="$NSYS_CMD $PROGRAM_CMD"
"""
    return LAUNCHER


def _make_deepspeed_launcher(args):

    if args.nsys:
        NSYS_CMD = _make_nsys_cmd(args)
    else:
        NSYS_CMD = ""
    LAUNCHER = f"""
# 2. Launcher
# Runs `deepspeed.launcher.launch` once on each node
# deepspeed.launcher sets the distributed env (RANK, WORLD_SIZE, etc.) and spawns processes for each local GPU
# additionally, it enables per rank logging -- this should be handled in config_parser and passed as part of
# the train_args
NSYS_CMD="{NSYS_CMD}"
PROGRAM_CMD="python -u -m deepspeed.launcher.launch \\
--node_rank \\$SLURM_PROCID \\
--master_addr $MASTER_NODE \\
--master_port $MASTER_PORT \\
\\$(<$TRAIN_ARGS_OUTPUT_PATH)"
export NSYS_CMD
export PROGRAM_CMD
LAUNCHER_CMD="$NSYS_CMD $PROGRAM_CMD"
"""
    return LAUNCHER


def _make_srun_launcher(args):
    if args.nsys:
        NSYS_CMD = _make_nsys_cmd(args)
    else:
        NSYS_CMD = ""
    
    if args.numa:
        NUMA_CMD = "numactl --cpunodebind=\\$((SLURM_LOCALID /4)) --membind=\\$((SLURM_LOCALID/4)) "
    else:
        NUMA_CMD = ""
    
    LAUNCHER = f"""
# 2. Launcher
# Runs `srun` launches world_size number of processes
# We need to set RANK and LOCAL_RANK manually
NSYS_CMD="{NSYS_CMD}"
PROGRAM_CMD="NCCL_DEBUG={args.nccl_debug.upper()} LOCAL_RANK=\\$SLURM_LOCALID RANK=\\$SLURM_PROCID {NUMA_CMD}python $TRAIN_SCRIPT \\$(<$TRAIN_ARGS_OUTPUT_PATH)"
export NSYS_CMD
export PROGRAM_CMD
LAUNCHER_CMD="$NSYS_CMD $PROGRAM_CMD"
"""

    return LAUNCHER

def _make_conditional_nsys(args):
    NSYS_STR = f""" 
# Select which nodes will collect nsys profiles
HOSTS=($(cat $HOSTLIST))
export HOSTS
STEP={args.profile_every}
HOST_SAMPLES=()
for ((i=0; i<${{#HOSTS[@]}}; i+=STEP)); do
    HOST_SAMPLES+=("${{HOSTS[i]}}")
done
HOST_SAMPLES_STR=$(echo "${{HOST_SAMPLES[@]}}")
export HOST_SAMPLES_STR

# Print the selected hosts
echo "nsys profiling nodes: ${{HOST_SAMPLES[@]}}"
export ROOT_DIR

srun $SRUN_ARGS bash -c '
curr_node=$(hostname)

if [[ " $HOST_SAMPLES_STR " =~ $curr_node ]]; then
    echo "$curr_node: sampling"
    CMD="export PYTHONPATH=$ROOT_DIR:$PYTHONPATH && $NSYS_CMD $PROGRAM_CMD"
else
    echo "$curr_node: not sampling"
    CMD="export PYTHONPATH=$ROOT_DIR:$PYTHONPATH && $PROGRAM_CMD"
fi

echo $CMD
eval $CMD
'
"""
    return NSYS_STR
def _generate_launcher_cmd(model_config: Path, args, CMDS):
    if args.launcher == "torch":
        LAUNCHER = _make_torch_launcher(args)
    elif args.launcher == "deepspeed":
        LAUNCHER = _make_deepspeed_launcher(args)
    elif args.launcher == "srun":
        LAUNCHER = _make_srun_launcher(args)
    else:
        raise ValueError(f"Unknown launcher type: {args.launcher}")

    if args.nsys and args.profile_every > 1:
        LAUNCHER += _make_conditional_nsys(args)
    else:
        # should_add_conda = args.pyxis and args.container == EVO_ONLY_CONTAINER
        extra_paths = "$ROOT_DIR"
        if args.use_next:
            extra_paths += ":$ROOT_DIR/nvidia-resiliency-ext/src"
        
        PYTHONPATH = f"{extra_paths}:$PYTHONPATH"
        BASE_CMD = f"export PYTHONPATH={PYTHONPATH} && ${{LAUNCHER_CMD}}"
        if not args.no_make_triton_cache:
            BASE_CMD = f"{MAKE_TRITON_CACHE_DIR}\n{BASE_CMD}"
        # if args.is_context_parallel:
        #     BASE_CMD = f"{INSTALL_CP_REQS} && \\\n{BASE_CMD}"
        LAUNCHER += f"""CMD="srun $SRUN_ARGS bash -c '{BASE_CMD}'"
"""

    if args.avoid_record_streams:
        LAUNCHER += f"""
export {RECORD_STREAMS_ENV_VAR}=1
"""
    if args.expandable_segments:
        LAUNCHER += f"""
export "{EXPANDABLE_SEGMENTS}"
"""
    if args.use_nvte_flash:
        LAUNCHER += f"""
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
#export NVTE_FUSED_ATTN_USE_FAv2_BWD=1
"""
    if args.nvte_debug:
        LAUNCHER += f"""
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
"""
    if args.disable_torch_compile:
        LAUNCHER += f"""
export NVTE_TORCH_COMPILE=0
"""
    if args.disable_pytorch_jit:
        LAUNCHER += f"""
export PYTORCH_JIT=0
"""

    LAUNCHER += """
echo $CMD
eval $CMD
"""

    CMDS += [LAUNCHER]
    return LAUNCHER, CMDS
