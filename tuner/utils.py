import copy
import datetime
import itertools
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

SAVANNA_RUN_VAR = "SAVANNA_RUN_ID"
WANDB_PROJECT_VAR = "WANDB_PROJECT"

def get_recursive_annotations(cls):
    annotations = {}
    # Traverse the class inheritance chain
    for base in cls.__mro__:
        if hasattr(base, '__annotations__'):
            annotations.update(base.__annotations__)
    return annotations


def update_config(config, param_combinations):
    """Update the YAML config with new parameters."""
    for param, value in param_combinations.items():
        keys = param.split(".")
        cfg = config
        for key in keys[:-1]:
            cfg = cfg[key]
        cfg[keys[-1]] = value
    return config


def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


def format_config_filename(param_combination):
    """Generate a unique filename based on the parameter values."""
    parts = []
    for param, value in param_combination.items():
        param_name = param.replace(".", "_")  # Replace dots in param names
        parts.append(f"{param_name}={value}")
    return "__".join(parts)


# Recursively update nested dictionaries
def recursive_update(d, u):
    for k, v in u.items():
        try:
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        except Exception as e:
            print(f"Error updating {k}: {e}")
    return d


def flatten_dict(d: dict, parent_key: str = "") -> dict:
    """
    Flattens a nested dictionary into a single-level dictionary.
    """

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Expand the parameters with multiple values into all combinations
def expand_search_space(search_space):
    flattened = flatten_dict(search_space)

    keys = []
    values = []
    updated_keys = []

    for key, val in flattened.items():
        if isinstance(val, list):
            keys.append(key)
            values.append(val)
            updated_keys.append(key)
        else:
            keys.append(key)
            values.append([val])

    # Generate all combinations of the multi-value parameters
    param_combinations = list(itertools.product(*values))

    expanded_configs = []
    for combo in param_combinations:
        config = {}
        for i, key in enumerate(keys):
            keys_hierarchy = key.split(".")
            d = config
            for part in keys_hierarchy[:-1]:
                d = d.setdefault(part, {})
            d[keys_hierarchy[-1]] = combo[i]
        expanded_configs.append(config)
    return expanded_configs, updated_keys


# Generate descriptive filename based on updated parameters
def generate_filename(base_name, updated_params):
    filename = base_name
    for param, value in updated_params.items():
        param_name = param.replace(".", "_")
        value_str = str(value).replace(".", "p")
        filename += f"__{param_name}={value_str}"
    return filename + ".yml"


# Generate new config files for each combination
def generate_configs(template_yaml, user_search_yaml, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Step 1: Expand the user-defined search space
    expanded_configs, param_keys = expand_search_space(user_search_yaml)

    # Step 2: Merge each expanded config with the template YAML
    config_count = 0
    config_paths = []

    for expanded_config in expanded_configs:
        # Merge expanded config with the template
        final_config = copy.deepcopy(template_yaml)
        recursive_update(final_config, expanded_config)

        flattened_config = flatten_dict(final_config)

        updated_params = {k: v for k, v in flattened_config.items() if k in param_keys}

        if len(updated_params) == 0:
            config_filename = f"config_{config_count}_{list(expanded_config.keys())[0]}.yml"
        else:
            config_filename = generate_filename(f"config_{config_count}", updated_params)

        config_filepath = os.path.join(output_dir, config_filename)

        with open(config_filepath, "w") as config_file:
            yaml.dump(final_config, config_file)

        config_count += 1
        config_paths.append(config_filepath)

    return config_paths


# Load YAML file
def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def print_delimiter(ch="-", length=80):
    print()
    print(ch * length)
    print()


@dataclass
class NsysConfig:
    enabled: bool = False
    gpu_metrics_device: str = "all"
    cuda_memory_usage: str = "true"
    cudabacktrace: str = "false"
    python_sampling: str = "false"
    capture_range: str = "cudaProfilerApi"
    stats: str = "false"
    nic_metrics: str = "true"
    show_output: str = "true"
    trace: str = "cuda,nvtx,osrt,cudnn,cublas-verbose"
    sample: str = "process-tree"
    force_overwrite: str = "true"
    stop_on_exit: str = "true"
    inherit_environment: str = "true"
    wait: str = "all"


def check_config(configs):
    from savanna.arguments import GlobalConfig

    global_config = GlobalConfig.from_ymls(configs)
    if global_config.hostfile is not None:
        raise ValueError("Running program only supports single node jobs, but a hostfile was provided.")


def _run_program(config_path, args, savanna_root):
    """Run the program with the specified YAML config."""
    # check_config([args.data_config, config_path])
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    wandb_project = args.wandb_project
    wandb_group = args.wandb_group
    if wandb_group is None:
        wandb_group = f"tune_{ts}"
    cmd = [
        "python",
        f"{savanna_root}/launch.py",
        f"{savanna_root}/train.py",
        args.data_config,
        config_path,
        "--wandb_project",
        wandb_project,
        "--wandb_group",
        wandb_group,
    ]
    print_delimiter()
    print(f"Running command: {' '.join(cmd)}")
    # Get the filename from config_path, removing parent directories and extension
    config_name = Path(config_path).stem
    os.environ[SAVANNA_RUN_VAR] = config_name

    subprocess.run(cmd, check=True, timeout=300)

    time.sleep(10)
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"])
    device_str = ", ".join(
        [f"Device {i}: {d}" for i, d in enumerate(result.decode("utf-8").strip().split("\n"))]
    )
    print(f"Memory used: {device_str}")