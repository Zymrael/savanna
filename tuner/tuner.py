import argparse
import datetime
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import yaml

import savanna
from savanna.arguments import GlobalConfig

TUNE_DIR = Path(__file__).resolve().parent
SAVANNA_DIR = TUNE_DIR.parent
import sys

sys.path.insert(0, str(SAVANNA_DIR))
from tuner.slurm_template import fill_slurm_template, fill_standalone_template
from tuner.utils import (
    NsysConfig,
    generate_configs,
    get_recursive_annotations,
    load_yaml,
)

DEFAULT_TEMPLATE_DIR = os.path.join(TUNE_DIR, "templates")
DEFAULT_DATA_PATH = os.path.join(TUNE_DIR, "data_configs/opengenome.yml")
DEFAULT_OUTPUT_DIR = os.path.join(TUNE_DIR, "generated_configs")
DEFAULT_NUM_NODES = 1
DEFAULT_NUM_GPUS = 1
DEFAULT_JOB_TIME = "00:10:00"

def calculate_gas(config, world_size):
    train_batch_size = config.get("train_batch_size")
    per_gpu_batch = config.get("train_micro_batch_size_per_gpu")
    mp_size = config.get("model_parallel_size", 1)
    pp_size = config.get("pipeline_parallel_size", 1)
    assert world_size % (mp_size * pp_size) == 0
    dp_size = world_size // (mp_size * pp_size)
    gas = train_batch_size // per_gpu_batch
    gas //= dp_size
    print("Calculated gas, gas = train_batch_size // (train_micro_batch_size_per_gpu * dp_size)\n"
          f"{gas} = {template_yaml['train_batch_size']} // ({template_yaml['train_micro_batch_size_per_gpu']} * {dp_size})")

    return gas
   
def check_config(configs):

    global_config = GlobalConfig.from_ymls(configs)
    if global_config.hostfile is not None:
        raise ValueError("Running program only supports single node jobs, but a hostfile was provided.")

def create_run_all_script(savanna_root, all_standalones_path, standalone_paths):
    with open(all_standalones_path, "w") as f:
        # Write the initial script headers
        f.write("#!/bin/bash\n")
        f.write(f"cd {savanna_root}\n")

        # Write commands to run each standalone script
        for standalone_path in standalone_paths:
            f.write(f"echo 'Running {standalone_path}...'\n")
            f.write(f"bash {standalone_path}\n")
            
            # Check for errors after each script execution
            f.write(dedent(f"""
                if [ $? -ne 0 ]; then
                    echo "Error: Failed to run {standalone_path}"
                else
                    echo "Success: {standalone_path} ran successfully"
                fi\n
            """))
        os.chmod(all_standalones_path, 0o755)
    print(f"Run all standalone scripts written at {all_standalones_path}")

def generate_scripts(
    job_name,
    savanna_root,
    configs_dir,
    config_paths,
    output_dir,
    wandb_project,
    wandb_group,
    nsys_config,
    log_dir=None,
    num_nodes=1,
    num_gpus=1,
    job_time="00:10:00",
    conda_env="evo2",
    data_config=DEFAULT_DATA_PATH,
    standalone_scripts=True,
):

    print("Generating scripts...")

    # Find all config files in the output directory
    config_files = list(Path(configs_dir).glob("*.yml"))
    num_configs = len(config_files)

    if num_configs == 0:
        print(f"Error: No config files found in {output_dir}")
        return

    print(f"Found {num_configs} config files.")

    # Create the SLURM job script
    slurm_script = fill_slurm_template(
        job_name=job_name,
        num_configs=num_configs,
        config_paths=config_paths,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        job_time=job_time,
        output_dir=output_dir,
        savanna_root=savanna_root,
        conda_env=conda_env,
        log_dir=log_dir,
        configs_dir=configs_dir,
        data_config=data_config,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        nsys_config=nsys_config,
    )

    job_script = os.path.join(output_dir, "slurm_job.sh")
    with open(job_script, "w") as f:
        f.write(slurm_script)

    print(f"SLURM job script created: {job_script}")

    if standalone_scripts:
        standalone_paths = []
        for config_path in config_paths:
            standalone_cmd_path = os.path.join(output_dir, f"{Path(config_path).stem}-standalone.sh")
            cmd = fill_standalone_template(
                data_config=data_config,
                model_config=config_path,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
                nsys_config=nsys_config,
                output_dir=output_dir,
            )
            with open(standalone_cmd_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"cd {savanna_root}\n")
                f.write(f"echo 'Running {standalone_cmd_path}:\n{cmd}'\n")
                f.write(cmd)
            os.chmod(standalone_cmd_path, 0o755)
            standalone_paths.append(standalone_cmd_path)
            print(f"Created standalone script: {standalone_cmd_path}")
        all_standalones_path = os.path.join(output_dir, "run_all_standalones.sh")
        create_run_all_script(savanna_root, all_standalones_path, standalone_paths)

    return job_script, standalone_paths if standalone_scripts else job_script


def submit_slurm_job_array(job_script, log_dir, wandb_project, wandb_group):
    # Submit the job array to SLURM
    submit_cmd = f"sbatch {job_script}"
    print(f"Submitting SLURM job array with: {submit_cmd}")

    result = subprocess.run(submit_cmd, shell=True)

    if result.returncode != 0:
        print("Error: Failed to submit SLURM job array.")
        return

    print(f"SLURM job array submitted successfully. Logging to {log_dir}")
    print(f"Run can be tracked on wandb at project/group: {wandb_project}/{wandb_group}")


# Function to convert the value to its correct type
def convert_value(value):
    if value.lower() == 'true':  # Check for booleans
        return True
    elif value.lower() == 'false':
        return False
    try:
        if '.' in value:  # Convert to float if the value contains a decimal
            return float(value)
        else:  # Convert to int if no decimal is present
            return int(value)
    except ValueError:
        return value  # If neither int nor float, return as string

def parse_key_value(s):
    try:
        key, value = s.split("=", 1)  # Only split on the first "="
        return key, convert_value(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not parse argument '{s}' as key=value")


def update_config_with_overrides(config, overrides):
    if overrides is None:
        return config
    print(f"Overrides: {type(overrides)} {overrides}")
    config_keys = dir(GlobalConfig)
    for k, v in overrides:
        if k in config:
            print(f"Overriding {k} with {v}")
            if type(config[k]) is not type(v):
                print(f"Warning: {k} is of type {type(config[k])} but {v} is of type {type(v)}. Converting to default type.")
                config[k] = type(config[k])(v)
            else:
                config[k] = v  
        else:
            if k not in config_keys:
                print(f"Warning: {k} not found in config. Skipping.")
                continue
            print(f"Adding `{k}`={v} to template")
            # Infer type of the value
            annotations = get_recursive_annotations(GlobalConfig)
            field_type = annotations[k]
            config[k] = field_type(v)
    return config

def cp_data_config(data_config: Path, output_dir):
    data_config_dir = os.path.join(output_dir, "data_configs")
    if not os.path.exists(data_config_dir):
        os.mkdir(data_config_dir)
    data_config_output_path = os.path.join(data_config_dir, data_config.name)
    # Copy the data config to the output dir
    shutil.copyfile(data_config, data_config_output_path)
    return data_config_output_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--search-config", type=str, default=None)
    parser.add_argument(
        "--template",
        type=Path,
        nargs="+",
        default=DEFAULT_TEMPLATE_DIR,
        help="Path to template file, can be a list",
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb_project", type=str, default="tuner")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true", help="Overwrite output directory")
    parser.add_argument("--submit-slurm", action="store_true", help="Generate configs and submit to SLURM")
    parser.add_argument(
        "--standalone-scripts", action="store_true", help="Generate standalone scripts (independent of SLURM)"
    )
    parser.add_argument("--calculate-gas", action="store_true", help="Calculate gas based on train_batch_size and" 
                        "train_micro_batch_size_per_gpu")
    # SLURM args
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--job-time", type=str, default="00:10:00")
    # nsys args
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--nsys_warmup_steps", type=int, default=3, help="Number of warmup steps before startin `nsys` profiler")
    parser.add_argument("--nsys_num_steps", type=int, default=2, help="Number of active `nsys` profiler steps")
    parser.add_argument(
        "--gpu-metrics-device", type=str, default="none", help="`nsys` flag: GPU metrics device"
    )
    parser.add_argument(
        "--cuda-memory-usage", type=str, default="true", help="`nsys` flag: CUDA memory usage flag"
    )
    parser.add_argument("--cudabacktrace", type=str, default="false", help="`nsys` flag: CUDA backtrace flag")
    parser.add_argument(
        "--python-sampling", type=str, default="false", help="`nsys` flag: Python sampling flag"
    )
    parser.add_argument(
        "--capture-range",
        type=str,
        default="cudaProfilerApi",
        help="`nsys` flag: Capture range for profiling",
    )
    parser.add_argument("--stats", type=str, default="false", help="`nsys` flag: Stats flag")
    parser.add_argument("--nic-metrics", type=str, default="true", help="`nsys` flag: NIC metrics flag")
    parser.add_argument("--show-output", type=str, default="true", help="`nsys` flag: Show output flag")
    parser.add_argument(
        "--trace",
        type=str,
        default="cuda,nvtx,osrt,cudnn,cublas-verbose",
        help="`nsys` flag: Trace options for nsys",
    )
    parser.add_argument("--sample", type=str, default="process-tree", help="`nsys` flag: Sampling method")
    parser.add_argument(
        "--force-overwrite", type=str, default="true", help="`nsys` flag: Force overwrite flag"
    )
    parser.add_argument("--stop-on-exit", type=str, default="false", help="`nsys` flag: Stop on exit flag")
    parser.add_argument(
        "--inherit-environment", type=str, default="true", help="`nsys` flag: Inherit environment flag"
    )
    parser.add_argument("--wait", type=str, default="all", help="`nsys` flag: Wait mode")
    # Key-value pairs to override the config
    parser.add_argument(
        "--overrides",
        nargs="*",  # Zero or more key=value pairs
        type=parse_key_value,
        default=None,
        help="Overrides in the form of key=value (e.g., learning_rate=0.01, model.hidden_size=1024)",
    )

    args = parser.parse_args()
    nsys_config = NsysConfig(
        enabled=args.nsys,
        gpu_metrics_device=args.gpu_metrics_device,
        cuda_memory_usage=args.cuda_memory_usage,
        cudabacktrace=args.cudabacktrace,
        python_sampling=args.python_sampling,
        capture_range=args.capture_range,
        stats=args.stats,
        nic_metrics=args.nic_metrics,
        show_output=args.show_output,
        trace=args.trace,
        sample=args.sample,
        force_overwrite=args.force_overwrite,
        stop_on_exit=args.stop_on_exit,
        inherit_environment=args.inherit_environment,
        wait=args.wait,
    )

    # Step 1: Parse the search space from grouped CLI args
    user_search_yaml = load_yaml(args.search_config) if args.search_config is not None else None
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    for template_path in args.template:
        # Generate the config files
        template_yaml = load_yaml(template_path)
        should_profile = template_yaml.get("should_profile", False)
        template_yaml = update_config_with_overrides(template_yaml, args.overrides)

        if args.calculate_gas:
            assert "train_batch_size" in template_yaml and "train_micro_batch_size_per_gpu" in template_yaml
            assert args.num_gpus is not None and args.num_nodes is not None
            world_size = args.num_gpus * args.num_nodes
            template_yaml["gradient_accumulation_steps"] = calculate_gas(template_yaml, args.num_gpus * args.num_nodes)
            
        if args.nsys:
            print(
                f"Overriding template nsys_num_steps with {args.nsys_num_steps} and nsys_num_warmup_steps with {args.nsys_warmup_steps}."
                " Please use commandline flags `--nsys_num_steps` and `--nsys_warmup_steps` to control profiling behavior rather than template"
                " model config when using `tuner.py`."
            )
            template_yaml["nsys_num_steps"] = args.nsys_num_steps
            template_yaml["nsys_warmup_steps"] = args.nsys_warmup_steps
            
        if not should_profile and args.nsys:
            print(
                f"WARNING: Template `{template_path}` does not have profiling enabled but `nsys` you specified `nsys` from the commandline."
                "  Changing template config to enable profiling."
            )
            template_yaml["should_profile"] = True
            template_yaml["profiler_type"] = "nsys"

        profiler_type = template_yaml.get("profiler_type", None)
        if should_profile and profiler_type != "nsys" and args.nsys:
            print(
                f"WARNING: Template `{template_path}` has profiler type `{profiler_type}` but you specified `nsys` from the commandline."
                "Changing profiler type to `nsys`."
            )
            template_yaml["profiler_type"] = "nsys"

        output_dir = os.path.join(args.output_dir, template_path.stem, ts)
        if args.overwrite_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)
        config_output_path = os.path.join(output_dir, "configs")

        if not os.path.exists(config_output_path):
            os.makedirs(config_output_path)

        if user_search_yaml is None:
            # in this case, just copy the template to the output dir
            config_paths = [os.path.join(config_output_path, os.path.basename(template_path))]

            with open(config_paths[0], "w") as f:
                yaml.dump(template_yaml, f)
        else:
            config_paths = generate_configs(template_yaml, user_search_yaml, config_output_path)

        config_paths_str = "\n  ".join(config_paths)
        print(
            f"Generated {len(config_paths)} config files for template `{template_path.stem}` and data config `{args.data_config.stem}`:\n  {config_paths_str}"
        )

        if args.wandb_group is None:
            wandb_group = template_path.stem #os.path.join(template_path.stem, ts)
        else:
            wandb_group = args.wandb_group

        savanna_dir = os.path.dirname(savanna.__file__)
        savanna_root = os.path.abspath(os.path.join(savanna_dir, os.pardir))

        log_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        data_config = cp_data_config(args.data_config, output_dir)
         
        slurm_script, standalone_scripts = generate_scripts(
            job_name=template_path.stem,
            configs_dir=config_output_path,
            config_paths=config_paths,
            data_config=data_config,
            output_dir=output_dir,
            log_dir=log_dir,
            num_nodes=args.num_nodes if args.num_nodes is not None else DEFAULT_NUM_NODES,
            num_gpus=args.num_gpus if args.num_gpus is not None else DEFAULT_NUM_GPUS,
            job_time=args.job_time if args.job_time is not None else DEFAULT_JOB_TIME,
            savanna_root=savanna_root,
            wandb_project=args.wandb_project,
            wandb_group=wandb_group,
            nsys_config=nsys_config,
        )
        print(f"Done generating scripts for template `{template_path.stem}`")

        if args.submit_slurm:
            submit_slurm_job_array(
                slurm_script, log_dir, wandb_project=args.wandb_project, wandb_group=wandb_group
            )