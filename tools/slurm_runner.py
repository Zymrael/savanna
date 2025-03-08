import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


# Function to update YAML model config
def update_model_config(model_config_path, config_updates):
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply user updates to config
    for key, value in config_updates.items():
        if key in config:
            print(f"Updating {key} from {config[key]} to {value}")
            config[key] = value
        else:
            print(f"Warning: {key} not found in {model_config_path}, skipping update")

    with open(model_config_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"Model config updated and saved to {model_config_path}")


# Function to create the SLURM script
def create_sbatch_script(args, timestamp, output_dir):
    sbatch_script = f"""#!/bin/bash

#SBATCH --job-name={args.job_name}       # Job name
#SBATCH --partition={args.partition}     # Partition to use
#SBATCH --nodes={args.nodes}             # Request nodes
#SBATCH --ntasks-per-node=1              # 1 task per node
#SBATCH --gres=gpu:{args.gpus_per_node}  # GPUs per node
#SBATCH --time={args.time}               # Time limit
#SBATCH --output={output_dir}/slurm_logs/%J_%A_%a-%N.out    # Output log
#SBATCH --error={output_dir}/slurm_logs/%J_%A_%a-%N.err     # Error log

# Set up output directory
OUTPUT_DIR="{output_dir}"
mkdir -p "$OUTPUT_DIR/slurm_logs"

# %h = hostname, %p = pid
OUTPUT_PREFIX="nsys_profile-%h-%p"

# Construct the nsys command
NSYS_CMD="nsys profile \\
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
--output=$OUTPUT_DIR/$OUTPUT_PREFIX \\
--force-overwrite={args.force_overwrite} \\
--stop-on-exit={args.stop_on_exit} \\
--inherit-environment={args.inherit_environment} \\
--wait={args.wait}"

PROGRAM_CMD="{args.python_exe} {args.program} {args.data_config} {args.model_config}"

MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
CURR_NODE=$(hostname)
NUM_GPUS=$SLURM_GPUS_ON_NODE

HOSTFILE="hostfile"
scontrol show hostname $SLURM_JOB_NODELIST > "$HOSTFILE"
sed -i "s/$/ slots=$NUM_GPUS/" "$HOSTFILE"

RUN_NSYS={1 if args.nsys else 0}
cd {os.getcwd()}
if [[ $RUN_NSYS -eq 1 ]]; then
    CMD="$NSYS_CMD $PROGRAM_CMD"
else
    CMD="$PROGRAM_CMD"
fi

echo "Running command: $CMD"
if [ "$CURR_NODE" = "$MASTER_NODE" ]; then
    $CMD
fi

if [[ $RUN_NSYS -eq 1 ]]; then
    echo "Saved traces to: $OUTPUT_DIR"
fi
"""
    return sbatch_script


# Function to submit the SLURM job
def submit_sbatch_script(sbatch_script, script_path):
    print("Submitting job with sbatch...")
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Job submitted successfully! Output: {result.stdout}")
    else:
        print(f"Error submitting job. Error: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic SLURM script generator for profiling")

    # Slurm arguments
    parser.add_argument("--job-name", type=str, default=None, help="SLURM job name")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to request")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--partition", type=str, default="gpu_batch", help="SLURM partition to use")
    parser.add_argument("--time", type=str, default="00:10:00", help="Job time limit")

    # Program and nsys options
    parser.add_argument(
        "--output-dir", type=str, default="slurm_runner_output", help="Output directory for traces and logs"
    )
    parser.add_argument("--python-exe", type=str, default="python", help="Python executable")
    parser.add_argument("--program", type=str, default="launch.py train.py", help="Program to run")
    parser.add_argument(
        "--data-config", type=str, default="tests/test_configs/minimal_data_config.yml", help="Data config file"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="tests/test_configs/profiler/e2e/single_node_1gpu_torch_profiler.yml",
        help="Model config file",
    )

    # nsys specific args
    parser.add_argument("--nsys", action="store_true", help="Enable nsys profiling")
    parser.add_argument("--gpu-metrics-device", type=str, default="none", help="GPU metrics device")
    parser.add_argument("--cuda-memory-usage", type=str, default="true", help="CUDA memory usage flag")
    parser.add_argument("--cudabacktrace", type=str, default="false", help="CUDA backtrace flag")
    parser.add_argument("--python-sampling", type=str, default="false", help="Python sampling flag")
    parser.add_argument(
        "--capture-range", type=str, default="cudaProfilerApi", help="Capture range for profiling"
    )
    parser.add_argument("--stats", type=str, default="false", help="Stats flag")
    parser.add_argument("--nic-metrics", type=str, default="true", help="NIC metrics flag")
    parser.add_argument("--show-output", type=str, default="true", help="Show output flag")
    parser.add_argument(
        "--trace", type=str, default="cuda,nvtx,osrt,cudnn,cublas-verbose", help="Trace options for nsys"
    )
    parser.add_argument("--sample", type=str, default="process-tree", help="Sampling method")
    parser.add_argument("--force-overwrite", type=str, default="true", help="Force overwrite flag")
    parser.add_argument("--stop-on-exit", type=str, default="true", help="Stop on exit flag")
    parser.add_argument("--inherit-environment", type=str, default="true", help="Inherit environment flag")
    parser.add_argument("--wait", type=str, default="all", help="Wait mode")
    # Additional options for updating model config
    parser.add_argument(
        "--config-updates", type=str, nargs="+", help="Key=value pairs to update in the model config YAML"
    )

    # debugging
    parser.add_argument("--dry-run", action="store_true", help="Print the SLURM script without submitting it")
    parser.add_argument(
        "--clean-output-dir", action="store_true", help="Clean the output directory before running the script"
    )

    args = parser.parse_args()
    args.job_name = args.job_name or str(Path(args.model_config).stem)
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = os.path.join(args.output_dir, args.job_name, timestamp)
    if args.clean_output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory created at: {output_dir}")
    print(f"Logs will be stored in: {output_dir}/slurm_logs/")

    # Update the model config YAML file if specified
    if args.config_updates:
        config_updates = dict(item.split("=") for item in args.config_updates)
        update_model_config(args.model_config, config_updates)

    # Create SLURM script
    sbatch_script = create_sbatch_script(args, timestamp, output_dir)

    # Write the script to a file and submit it
    script_path = os.path.join(output_dir, "slurm_script.sh")

    with open(script_path, "w") as f:
        f.write(sbatch_script)

    print(f"SLURM script written to {script_path}")

    if not args.dry_run:
        submit_sbatch_script(sbatch_script, script_path)


if __name__ == "__main__":
    main()
