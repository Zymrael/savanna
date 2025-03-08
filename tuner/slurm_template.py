import datetime
import random
from pathlib import Path

from tuner.utils import NsysConfig


def generate_nsys_cmd(nsys_config: NsysConfig, output_dir: str, prefix: str="nsys-%h-%p"):
    NSYS_CMD=f"""nsys profile \\
    --gpu-metrics-device={nsys_config.gpu_metrics_device} \\
    --cuda-memory-usage={nsys_config.cuda_memory_usage} \\
    --cudabacktrace={nsys_config.cudabacktrace} \\
    --python-sampling={nsys_config.python_sampling} \\
    --capture-range={nsys_config.capture_range} \\
    --stats={nsys_config.stats} \\
    --nic-metrics={nsys_config.nic_metrics} \\
    --show-output={nsys_config.show_output} \\
    --trace={nsys_config.trace} \\
    --sample={nsys_config.sample} \\
    --output={output_dir}/{prefix} \\
    --force-overwrite={nsys_config.force_overwrite} \\
    --stop-on-exit={nsys_config.stop_on_exit} \\
    --inherit-environment={nsys_config.inherit_environment} \\
    --wait={nsys_config.wait} \\"""
    return NSYS_CMD
    
def generate_program_cmd(data_config, wandb_project, wandb_group, model_config="$CONFIG_FILE", run_id="$RUN_ID"):
    return f"""python launch.py train.py \\
    {data_config} {model_config} \\
    --wandb_project {wandb_project} \\
    --wandb_group {wandb_group} \\
    --wandb_run_name {run_id}"""

def fill_standalone_template(*, data_config, model_config, wandb_project, wandb_group, nsys_config: NsysConfig, output_dir: str):
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    cmd = generate_program_cmd(data_config, wandb_project, wandb_group, model_config=model_config, run_id=f"{Path(model_config).stem}-{ts}")
    if nsys_config.enabled:
        nsys_cmd = generate_nsys_cmd(nsys_config, output_dir, prefix=Path(model_config).stem + "-%h-%p")
        cmd = f"{nsys_cmd}\n\t{cmd}"
    return cmd

def fill_slurm_template(
    *,
    job_name,
    num_configs,
    config_paths,
    num_nodes,
    num_gpus,
    job_time,
    savanna_root,
    output_dir,
    conda_env,
    log_dir,
    configs_dir,
    data_config,
    wandb_project,
    wandb_group,
    nsys_config,
):
    # Set up job array only if num_configs > 1
    if num_configs > 1:
        assert len(config_paths) > 1
        array_str = f"#SBATCH --array=0-{num_configs - 1}"
        config_file_selection = f'CONFIG_FILE=$(ls {configs_dir}/*.yml | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")'
        run_id_selection = 'RUN_ID="${file_base%.*}"'
        log_suffix = "%A_%a"
    else:
        assert len(config_paths) == 1
        array_str = ""  # No job array
        config_file = config_paths[0]
        config_file_selection = f'CONFIG_FILE={config_file}'
        run_id_selection = f'RUN_ID={Path(config_file).stem}'
        log_suffix = "%J"
    MASTER_PORT = random.randint(10000, 65535)

    # Base SLURM job setup
    SETUP = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{num_gpus}
{array_str}
#SBATCH --output={log_dir}/slurm-%J-{log_suffix}-%N.out
#SBATCH --error={log_dir}/slurm-%J-{log_suffix}-%N.err
#SBATCH --time={job_time}
#SBATCH --partition=gpu_batch

# Activate conda environment
#eval "$(conda shell.bash hook)" && conda activate {conda_env}

cd {savanna_root}

MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
CURR_NODE=$(hostname)

if [ "$CURR_NODE" = "$MASTER_NODE" ]; then
    scontrol show hostname $SLURM_JOB_NODELIST > {savanna_root}/hostfile
    sed -i "s/$/ slots={num_gpus}/" {savanna_root}/hostfile
fi

# Select the config file
{config_file_selection}

# Extract the base filename without the directory and extension
file_base=$(basename $CONFIG_FILE)
{run_id_selection}
TS="$(date +%Y%m%d%H%M)"
RUN_ID=$RUN_ID-$TS

# Prevent Host key verification errors
export PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
MASTER_PORT=$(({MASTER_PORT} + ${'{SLURM_ARRAY_TASK_ID:-0}'}))

echo "Running job with config file: $CONFIG_FILE, WANDB project: {wandb_project}, WANDB group: {wandb_group}, run ID: $RUN_ID"

export MASTER_PORT=$MASTER_PORT
export SAVANNA_RUN_ID=$RUN_ID"""

    # NSYS command generation
    NSYS_CMD = generate_nsys_cmd(nsys_config, output_dir, prefix="$RUN_ID-%h-%p")     

    # Program command generation
    PROGRAM_CMD = generate_program_cmd(data_config, wandb_project, wandb_group, model_config="$CONFIG_FILE")

    # Final command block, with or without NSYS depending on configuration
    if nsys_config.enabled:
        CMD = f"""{NSYS_CMD}
    {PROGRAM_CMD}"""
    else:
        CMD = f"""{PROGRAM_CMD}"""

    RUN_PROGRAM_CMD = f"""
CMD="{CMD}"

echo "Running command: $CMD"

if [ "$CURR_NODE" = "$MASTER_NODE" ]; then
    $CMD
fi
"""

    return "\n".join([SETUP, RUN_PROGRAM_CMD])
