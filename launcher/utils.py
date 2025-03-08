import argparse
import itertools
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

#import yaml
# try to import ruamel.yaml if available else fallback to yaml
try:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    
except ImportError:
    print("ruamel.yaml not found, falling back to yaml")
    import yaml

from launcher.constants import (
    DEFAULT_ACCOUNT,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONTAINER_MOUNTS,
    DEFAULT_CONTAINER_WORKDIR,
    DEFAULT_DATA_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARTITION,
    DEFAULT_WANDB_HOST,
    EFS_EVO_CONTAINER,
    LAUNCHER_DIR,
    SAVANNA_ROOT,
)


def _get_default_output_dir():
    user = os.environ.get("USER", None)
    assert user is not None and user != "root", "User {user} not valid, please provide output directory"
    return Path(DEFAULT_OUTPUT_DIR) / user


def _generate_new_config(template, config_updates):
    config = deepcopy(template)
    for k, v in config_updates.items():
        if k not in template:
            print(f"Warning: {k} not found in config. Skipping.")
            continue
        config[k] = v

    return config


def _generate_config_str(config: dict):
    config_str = "_".join([f"{k}={v}" for k, v in config.items()])
    return config_str


def save_configs(configs: list, search_combos: list, args):

    generated_config_dir = args.output_dir / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    config_paths = []
    for config, search_combo in zip(configs, search_combos):
        search_combo_str = _generate_config_str(search_combo)
        save_path = generated_config_dir / f"{search_combo_str}.yml"
        with open(save_path, "w") as f:
            yaml.dump(config, f)
        config_paths.append(save_path)
    print(f"Wrote {len(configs)} search configs to {generated_config_dir}")
    args.generated_configs_dir = generated_config_dir
    return config_paths


def generate_new_configs(template: Path, search_config: Path):
    """
    Given a template and a search config, generates all possible combinations
    of the template for search config values.
    """
    base_config = load_yaml(template)
    search_space = load_yaml(search_config)
    # Generate all possible combinations of search config values
    search_keys = set(search_space.keys())
    search_vals = itertools.product(*search_space.values())
    search_combos = [dict(zip(search_keys, v)) for v in search_vals]

    generated_configs = []
    for combo in search_combos:
        config = _generate_new_config(base_config, combo)
        generated_configs.append(config)

    return generated_configs, search_combos


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.load(f) #, Loader=yaml.FullLoader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate launcher SLURM script launching distributed training jobs without `pdsh`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Launcher parameters
    parser.add_argument(
        "--launcher",
        type=str,
        default="srun",
        choices=["torch", "deepspeed", "srun"],
        help="Type of launcher",
    )
    # SLURM parameters
    parser.add_argument("job_name", type=str, help="`SLURM` - name of the SLURM job.")
    parser.add_argument("--account", type=str, default=DEFAULT_ACCOUNT, help="`SLURM` - account to use.")
    parser.add_argument("--partition", type=str, default=DEFAULT_PARTITION, help="`SLURM` - partition")
    parser.add_argument("--num-nodes", type=int, default=2, help="`SLURM` -number of nodes.")
    parser.add_argument("--num-gpus", type=int, default=8, help="`SLURM` - number of GPUs per node.")
    parser.add_argument("--cpus-per-task", type=int, default=None, help="`SLURM` - number of CPUs per task.")
    parser.add_argument("--mem", type=str, default="0", help="`SLURM` - memory per node.")
    parser.add_argument("--exclusive", action="store_true", help="`SLURM` - exclusive node usage")
    parser.add_argument("--job-time", type=str, default="01:00:00", help="`SLURM` - job time limit.")
    parser.add_argument(
        "--pyxis", action="store_true", default=True, help="use `pyxis` to run containerized SLURM jobs."
    )
    parser.add_argument("--job-array", action="store_true", default=False, help="use SLURM job array.")
    parser.add_argument(
        "--no-pyxis",
        dest="pyxis",
        action="store_false",
        help="disable pyxis (i.e., `srun` in no container mode).",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=EFS_EVO_CONTAINER,
        help="`pyxis` - container image",
    )
    parser.add_argument(
        "--container-mounts", type=str, default=DEFAULT_CONTAINER_MOUNTS, help="`pyxis` - container mounts."
    )
    parser.add_argument(
        "--container-workdir",
        type=str,
        default=DEFAULT_CONTAINER_WORKDIR,
        help="`pyxis` - container workdir.",
    )
    parser.add_argument(
        "--mount-home",
        action="store_true",
        help="`pyxis` - mount home dir, default is to NOT mount home dir.",
    )

    # Training script parameters
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Location of evo2 data.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store logs.")
    parser.add_argument(
        "--root",
        type=str,
        default=SAVANNA_ROOT,
        help="Root directory where `savanna` repo is located. E.g., `/home/$USER/savanna`.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=SAVANNA_ROOT / "train.py",
        help="Training script name, most likely `savanna/train.py`.  Note: ensure `savanna` root is mapped correctly in the container.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to data config.  Note: ensure the location is relative to location in the *container*.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to model config.  Note: ensure the location is relative to location in the *container*.",
    )
    parser.add_argument(
        "--hostlist",
        type=str,
        default="hostlist",
        help="Hostlist file name where SLURM host names are stored.  Primarily for debugging.",
    )
    parser.add_argument(
        "--train-args-output", type=str, default="train_args.txt", help="Output file for parsed train args."
    )
    parser.add_argument(
        "--config-parser",
        type=str,
        default=LAUNCHER_DIR / "config_parser.py",
        help="Config parser script name.  Only change this if debugging.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for logs and other artifacts, defaults to job name.",
    )
    
    parser.add_argument("--use-wandb", action="store_true", help="Use `wandb` for logging.")
    parser.add_argument("--wandb-host", type=str, default=DEFAULT_WANDB_HOST, help="Wandb host.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--search-config", type=Path, default=None, help="Path to yml search config.")
    parser.add_argument("--hpz_partition_size", type=int, default=None, help="hpZ partition size")
    parser.add_argument(
        "--overrides",
        nargs="*",  # Zero or more key=value pairs
        type=parse_key_value,
        default=None,
        help="Overrides in the form of key=value (e.g., learning_rate=0.01, model.hidden_size=1024)",
    )
   # parser.add_argument("--triton_cache", type=Path, default=None, help="Path to triton cache.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="checkpoint directory"
    )
    parser.add_argument("--checkpoint_path", type=Path, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--iteration", type=int, default=None, help="Checkpoint iteration to resume from.")
    parser.add_argument("--disable-checkpoint", action="store_true", help="Disable checkpointing / model saving, for debugging")
    # Profiling
    parser.add_argument("--nsys", action="store_true", help="Enable `nsys` profiling.")
    parser.add_argument(
        "--profile_every",
        type=int,
        default=1,
        help="Sample from every `profile_every` hosts, e.g., if 1, then all hosts will run nsys, 2 every other host will run nsys.",
    )
    parser.add_argument(
        "--nsys_warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps before startin `nsys` profiler",
    )
    parser.add_argument(
        "--nsys_num_steps", type=int, default=1, help="Number of active `nsys` profiler steps"
    )
    parser.add_argument(
        "--gpu-metrics-device", type=str, default="none", help="`nsys` flag: GPU metrics device"
    )
    parser.add_argument(
        "--cuda-memory-usage", type=str, default="false", help="`nsys` flag: CUDA memory usage flag"
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
    parser.add_argument("--nic-metrics", type=str, default="false", help="`nsys` flag: NIC metrics flag")
    parser.add_argument("--show-output", type=str, default="true", help="`nsys` flag: Show output flag")
    parser.add_argument(
        "--trace",
        type=str,
        default="cuda,nvtx",  # ,osrt,cudnn,cublas-verbose
        help="`nsys` flag: Trace options for nsys",
    )
    parser.add_argument("--sample", type=str, default="none", help="`nsys` flag: Sampling method")
    parser.add_argument(
        "--force-overwrite", type=str, default="true", help="`nsys` flag: Force overwrite flag"
    )
    parser.add_argument("--stop-on-exit", type=str, default="true", help="`nsys` flag: Stop on exit flag")
    parser.add_argument(
        "--wait", type=str, default="all", choices=["primary", "all"], help="`nsys` flag: Wait flag"
    )

    # Torch profiler
    parser.add_argument("--torch_profiler", action="store_true", help="Enable `torch_profiler` profiler.")

    # deepspeed launcher options
    parser.add_argument(
        "--enable-each-rank-log",
        action="store_true",
        help="Enable each rank log when using deepspeed launcher.",
    )
    # torchrun params
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Master port for distributed training, automatically generated if not specified.",
    )
    parser.add_argument("--rdzv_backend", type=str, default="c10d", help="`torchrun` - rdzv backend")
    parser.add_argument("--max-restarts", type=int, default=0, help="`torchrun` - maximum number of restarts")

    # Experimental
    parser.add_argument(
        "--avoid_record_streams",
        action="store_true",
        help="Enable `TORCH_NCCL_AVOID_RECORD_STREAMS` env var.",
    )
    parser.add_argument(
        "--expandable_segments",
        action="store_true",
        help="Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var.",
    )

    # Misc
    parser.add_argument("--verbose", action="store_true", help="Prints script to stdout.")
    parser.add_argument("--enable-heimdall", action="store_true", help="Enable Heimdall monitoring.")
    parser.add_argument("--no_make_triton_cache", action="store_true", help="Do not create process-specific triton cache directory.")

    # nvidia resiliency ext
    parser.add_argument("--use-next", action="store_true", help="Enable nvidia-resiliency-ext")
    
    # heimdall straggler detection
    parser.add_argument("--heimdall_log_straggler", action="store_true", help="Enable heimdall straggler detector")
    parser.add_argument("--heimdall_log_interval", type=int, default=10, help="Heimdall log interval")
    parser.add_argument("--heimdall_straggler_minmax_count", type=int, default=32)
    parser.add_argument("--heimdall_straggler_port", type=int, default=65535)
    
    # async saving
    parser.add_argument("--numa", action="store_true", help="Enable NUMA process binding")
    parser.add_argument("--enable_async_save", action="store_true", help="Enable async saving")
    
    # transformer engine
    parser.add_argument("--use-nvte-flash", action="store_true", help="Enable transformer engine flash attention")
    parser.add_argument("--nvte-debug", action="store_true", help="Enable transformer engine debug")
    parser.add_argument("--recycle-events", action="store_true", help="Enable event recycling")

    parser.add_argument("--nccl-debug", type=str, default="WARN", help="NCCL_DEBUG level")

    args = parser.parse_args()
    
    # Check train script is relative to root
    train_script = Path(args.train_script).absolute()
    assert train_script.is_relative_to(SAVANNA_ROOT), "Train script must be relative to root"
    return args


def convert_value(value):
    if isinstance(value, str):  # Convert strings to booleans
        if value.lower() == "true":  # Check for booleans
            return True
        elif value.lower() == "false":
            return False
        elif value.startswith("[") and value.endswith("]"):  # Check for lists
            return [convert_value(v) for v in value[1:-1].split(",")]
        try:
            if "." in value:  # Convert to float if the value contains a decimal
                return float(value)
            else:  # Convert to int if no decimal is present
                return int(value)
        except ValueError:
            return value  # If neither int nor float, return as string
    else:
        print(f"No conversion for type {type(value)} {value}")
        return value  # If not a string, return as is


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
    config_keys = list(config.keys())
    for k, v in overrides:
        if k in config:
            print(f"  Overriding {k} with {v}")
            if type(config[k]) is not type(v):
                print(
                    f"Warning: {k} is of type {type(config[k])} but {v} is of type {type(v)}. Converting to default type."
                )
                config[k] = type(config[k])(v)
            else:
                config[k] = v
        else:
            if k not in config_keys:
                print(
                    f"Warning: {k} not found in config: make sure that this is a valid key else will result in downstream errors."
                )
            # Values should were already converted to types during arg parsing
            config[k] = v
            print(f"Adding `{k}`={v} to template")

    return config


def create_checkpoint_dir(args, subdir):
    if args.checkpoint_path is None:
        checkpoint_dir = os.path.join(DEFAULT_CHECKPOINT_DIR, args.job_name, subdir, args.time_stamp)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        args.checkpoint_path = checkpoint_dir
    return args.checkpoint_path

def configure_checkpoint_stores(model_config, model_config_path, args):
    # Autoconfigure checkpoint store
    # assert model_config.get("save", None) is None, "Model config should not have save in config when using script generator"
    if args.resume:
        assert model_config.get("save", None) is not None, "Model config should have save path already set when resuming training"
    else:    
        checkpoint_dir = create_checkpoint_dir(args, model_config_path.stem)
        model_config["save"] = str(checkpoint_dir)
        
        # if model_config.get("load", None) is None:
        #     model_config["load"] = str(checkpoint_dir)

        print(
            f"""Setting `save` to {checkpoint_dir}"""# and `load` to: {model_config["load"]}.\n  -> When restarting training, this ensures that the model will be loaded from and saved to same original directory."""
        )
        #         print(
        #     f"""Setting `save` to {checkpoint_dir} and `load` to: {model_config["load"]}.\n  -> When restarting training, this ensures that the model will be loaded from and saved to same original directory."""
        # )

    checkpoint_store = model_config.get("checkpoint-stores", None)
    if checkpoint_store is not None:
        # save_path = create_checkpoint_dir(args, model_config_path.stem)
        # model_config["save"] = save_path
        # print(
        #     f"Setting `save` path in model config to {save_path} to ensure local checkpoint is stored on lustre"
        # )

        assert len(checkpoint_store) == 1, "Only one checkpoint store is supported"
        remote_config = checkpoint_store[0]
        if remote_config.get("location", None) is None:
            # subdir = save_path.split("/")[-2:]
            remote_path = os.path.join("chkpt", args.job_name, model_config_path.stem)
            remote_config["location"] = remote_path
            print(f"Setting remote store location to {remote_path}")

def update_config(model_config_path: Path, args):
    # Copy data and model config to output path for reproducibility
    output_path = args.output_dir

    # Move profiler outputs to output_path
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f)

    # overrides
    model_config = update_config_with_overrides(model_config, args.overrides)
    print_delimiter(num_chars=50)

    # Checkpointing
    # If we're resuming, the checkpoint directory should be provided already and we should set "save" to the 
    # provided checkpoint path to ensure that new checkpoints are saved to the same checkpoint directory as previous checkpoints
    # if not, we either a) disable checkpointing, in which case no need to create checkpoint path
    # or b) create a new checkpoint path
    
    if args.resume:
        # Check checkpoint load iteration
        iteration = model_config.get("iteration", None) or args.iteration
        assert (
            iteration is not None
        ), "`iteration` must be specified in model config or by commandline when resuming training"
        
        model_config["iteration"] = iteration
        args.iteration = iteration
        
        # Update checkpoint path
        checkpoint_path = model_config.get("load", None) or args.checkpoint_path

        assert checkpoint_path is not None, "`checkpoint_path` must be specified in model config or by commandline when resuming training"
        assert os.path.exists(
                checkpoint_path
            ), f"Checkpoint path {args.checkpoint_path} does not exist"

        if model_config.get("load", None) is not None and model_config["load"] != str(args.checkpoint_path):
            print(
                    f"WARNING: model config load != args.checkpoint_path: {model_config['load']} != {args.checkpoint_path}"
                )

        model_config["load"] = str(checkpoint_path)
        args.checkpoint_path = str(checkpoint_path)
        
        # Ensure new checkpoints are saved to same directory as before 
        model_config["save"] = str(checkpoint_path)        
        
        # Check that checkpoint-stores exists
        # Ensure lr and index are correctly restored
        model_config["use_checkpoint_lr_scheduler"] = True
        model_config["use_checkpoint_num_samples"] = True        
        print(f"Resuming training with checkpointing settings:\n  `iteration` {model_config['iteration']}\n  `load`: {checkpoint_path}\n  `use_checkpoint_lr_scheduler=True`")
    else:
        if args.disable_checkpoint:
            if model_config.get("checkpoint-stores", None) is not None:
                del model_config["checkpoint-stores"]
            # if model_config.get("save", None) is not None:
            #     del model_config["save"]
            print("Disabling checkpointing")
        else:
            configure_checkpoint_stores(model_config, model_config_path, args)    
    
    # These should be always be set as such when using this tool
    if not model_config.get("use_srun_launcher", False):
        print("Setting use_srun_launcher True in model config.")
        model_config["use_srun_launcher"] = True

    # Set launcher type based on CLI
    print(f"Setting launcher type to {args.launcher}")
    model_config["srun_launcher_type"] = args.launcher

    # Remove incompatible args
    if "master_port" in model_config:
        del model_config["master_port"]
    if "hostfile" in model_config:
        del model_config["hostfile"]
    if "launcher" in model_config:
        del model_config["launcher"]
    if "deepspeed_slurm" in model_config:
        del model_config["deepspeed_slurm"]

    # Update config based on CLI args
    # deepspeed launcher
    if args.enable_each_rank_log:
        model_config["enable_each_rank_log"] = True

    # wandb
    use_wandb = model_config.get("use_wandb", False) or args.use_wandb
    if use_wandb:
        check_wandb(args.wandb_host)
        wandb_project, wandb_group, wandb_run_name = configure_wandb(args, model_config_path)
        model_config["use_wandb"] = True
        model_config["wandb_project"] = wandb_project
        model_config["wandb_group"] = wandb_group
        model_config["wandb_run_name"] = wandb_run_name
        print(f"Setting wandb project/group/run name to {wandb_project}/{wandb_group}/{wandb_run_name}")

    # Profiler updates
    should_profile = model_config.get("should_profile", False)
    profiler_type = model_config.get("profiler_type", None)
    use_nsys = profiler_type == "nsys" or args.nsys

    assert not (
        args.nsys and args.torch_profiler
    ), "nsys and torch_profiler cannot be enabled at the same time."
    # Torch profiler
    if (should_profile and profiler_type == "torch") or args.torch_profiler:
        profiler_output_dir = output_path / "torchprofiler_traces"
        profiler_output_dir.mkdir(parents=True, exist_ok=True)
        model_config["should_profile"] = True
        model_config["profiler_type"] = "torch"

        model_config["profiler_output_dir"] = str(profiler_output_dir)
        world_size = args.num_nodes * args.num_gpus

        if args.profile_every == -1:
            model_config["profile_ranks"] = list(range(0, world_size))
        if args.profile_every is None:
            if args.num_nodes >= 4:
                # Profile once per node
                model_config["profile_ranks"] = list(range(0, world_size, args.num_gpus))
            else:
                model_config["profile_ranks"] = list(range(0, world_size))
        else:
            model_config["profile_ranks"] = list(range(0, world_size, args.profile_every))
        print(
            f"Setting profiler type to torch_profiler. Outputs written to {profiler_output_dir}. Setting profile_ranks to {model_config['profile_ranks']}"
        )

    elif use_nsys:
        model_config["should_profile"] = True
        model_config["profiler_type"] = "nsys"
        model_config["nsys_num_steps"] = args.nsys_num_steps
        model_config["nsys_warmup_steps"] = args.nsys_warmup_steps
        print("Setting profiler type to nsys")

    if model_config.get("print_mem_alloc_stats", False):
        # Limit number of printing based on number of ranks
        if args.num_nodes < 4:
            # Every rank prints
            model_config["mem_alloc_stats_ranks"] = 1
        else:
            # Sample 2 ranks per node
            model_config["mem_alloc_stats_ranks"] = 4

        print(f"Setting mem_alloc_stats_ranks to {model_config['mem_alloc_stats_ranks']}")

    zero_opt = model_config.get("zero_optimization", None)
    if zero_opt is not None and args.hpz_partition_size is not None:
        zero_opt["zero_hpz_partition_size"] = args.hpz_partition_size
        print(f"Setting zero_hpz_partition_size to {args.hpz_partition_size}")

    # For 40b training
    if args.expandable_segments:
        model_config['expandable_segments'] = True
        print("Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    should_use_next = model_config.get("use_next", False) or args.use_next
    if should_use_next:
        model_config['use_next'] = True
        args.use_next = True
        print("Setting `use_next` to True")
        next_path = Path(__file__).parent.parent / 'nvidia-resiliency-ext'
        assert next_path.exists(), "Next path {} does not exist".format(next_path)

    should_use_hsd = model_config.get("heimdall_log_straggler", False) or args.heimdall_log_straggler
    if should_use_hsd:
        model_config['heimdall_log_straggler'] = True
        args.heimdall_log_straggler = True
        hsd_log_interval = model_config.get("heimdall_log_interval", args.heimdall_log_interval)
        model_config['heimdall_log_interval'] = hsd_log_interval
        hsd_straggler_port = model_config.get("heimdall_straggler_port", args.heimdall_straggler_port)
        model_config['heimdall_straggler_port'] = hsd_straggler_port
        hsd_minmax_cnt = model_config.get("heimdall_straggler_minmax_count", args.heimdall_straggler_minmax_count)
        model_config["heimdall_straggler_minmax_count"] = hsd_minmax_cnt
        model_config["heimdall_disable_straggler_on_startup"] = False
        print("Heimdall straggler configuration:")
        print("  log interval: {}".format(hsd_log_interval))
        print("  straggler port: {}".format(hsd_straggler_port))
        print("  minmax count: {}".format(hsd_minmax_cnt))
        
    if args.enable_async_save or model_config.get("async_save", False):
        model_config["async_save"] = True
        print("Setting async_save to True")
        args.numa = True
    
    # Temporary fix needed for CP before these requirements are baked into the container
    if model_config.get("context_parallel_size", None) is not None and model_config["context_parallel_size"] >= 2:
        args.is_context_parallel = True
        print("Setting is_context_parallel to True")
    else:
        args.is_context_parallel = False
        print("Setting is_context_parallel to False")
        
    if args.recycle_events or model_config.get("recycle_events", False):
        model_config["recycle_events"] = True
        print("Setting recycle_events to True")
    else:
        model_config["recycle_events"] = False
        print("Setting recycle_events to False")

    if model_config["recycle_events"] and model_config.get("use_cp_flash_te", False):
        args.disable_torch_compile = True
        print("Disabling torch compile since recycling events and using TE Attn")
        args.disable_pytorch_jit = True
        print("Disabling pytorch jit since recycling events and using TE Attn")
    else:
        args.disable_torch_compile = False
        args.disable_pytorch_jit = False
        
    return model_config


def set_master_port(args):
    if args.master_port is None:
        args.master_port = random.randint(10000, 20000)

    return args


def get_timestamp(fmt="%Y%m%d%H%M"):
    return datetime.now().strftime(fmt)


def prep_paths(args):
    """
    Prep output directory

    Copy all configs as well as redirect all outputs to the output directory.
    """

    ts = get_timestamp()
    args.time_stamp = ts

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _get_default_output_dir()
        output_dir = output_dir / args.job_name
    print(f"All outputs will be stored in {output_dir / ts}")

    output_path = output_dir / ts
    output_path.mkdir(parents=True, exist_ok=True)

    # update output_dir in place
    args.output_dir = output_path

    # make log dir
    log_dir = args.output_dir / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logs will be generated in {log_dir}")

    # Set triton cache
    # triton_cache = args.triton_cache
    # if triton_cache is None:
    #     triton_cache = output_path / DEFAULT_TRITON_CACHE
    #     triton_cache.mkdir(parents=True, exist_ok=True)
    #     print(f"Triton cache will be stored in {triton_cache}")

    # args.triton_cache = triton_cache

    return output_path


def check_wandb(wandb_host):
    # First check if user exported WANDB_API_KEY
    wandb_token = os.environ.get("WANDB_API_KEY", None)
    assert (
        wandb_token is not None
    ), "Wandb token not found, please export WANDB_API_KEY (e.g.,. export WANDB_API_KEY=my_api_key)"


def configure_wandb(args, model_config: Path):
    # Check first config to see if wandb is enabled
    wandb_project = args.wandb_project or args.job_name
    wandb_group = args.wandb_group or get_timestamp()
    wandb_run_name = args.wandb_run_name or model_config.stem

    args.wandb_project = wandb_project
    args.wandb_group = wandb_group
    args.wandb_run_name = wandb_run_name

    return wandb_project, wandb_group, wandb_run_name


def get_model_configs(args):
    model_config: Path = args.model_config
    if model_config.is_dir():
        model_configs = list(model_config.glob("*.yml"))
        assert len(model_configs) > 0, f"No model configs found in {model_config}"
    else:
        model_configs = [model_config]

    return model_configs


def check_data_config(data_config: Path):
    with open(data_config) as f:
        data_config = yaml.load(f)

    if "use_wandb" in data_config:
        raise ValueError(
            "use_wandb is not supported in data config. Please either move it to the model config or specify `--use-wandb`."
        )


def transfer_data_config(data_config: Path, args):
    output_path = args.output_dir
    data_config_dir = output_path / "data_config"
    data_config_dir.mkdir(parents=True, exist_ok=True)
    data_config_path = data_config_dir / args.data_config.name
    args.data_config_dir = data_config_dir

    shutil.copyfile(data_config, data_config_path)

    print(f"Data config saved at {data_config_dir}")

    return data_config_path


def transfer_model_configs(model_config_paths: List[Path], model_configs: List[dict], args):
    output_path = args.output_dir
    model_configs_dir = output_path / "model_configs"
    model_configs_dir.mkdir(parents=True, exist_ok=True)
    updated_model_config_paths = []
    config_names = [p.name for p in model_config_paths]

    for config_name, model_config in zip(config_names, model_configs):
        model_config_path = model_configs_dir / config_name
        with open(model_config_path, "w") as f:
            yaml.dump(model_config, f)
        updated_model_config_paths.append(model_config_path)
    print(f"Model configs saved at {model_configs_dir}")

    return model_configs_dir, updated_model_config_paths


def update_model_configs(model_configs, args):
    return [update_config(model_config, args) for model_config in model_configs]


def make_submit_script(script_dir: Path):
    submit_path = script_dir / "submit.sh"
    template = """
#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all .sbatch files in the script directory and run them with sbatch
for sbatch_file in "$SCRIPT_DIR"/*.sbatch; do
    if [[ -f "$sbatch_file" ]]; then
        echo "Submitting $sbatch_file..."
        sbatch "$sbatch_file"
    fi
done
"""

    with open(submit_path, "w") as f:
        f.write(template)

    os.chmod(submit_path, 0o777)
    print(f"Submit script saved at {submit_path}")
    return submit_path

def print_delimiter(ch="-", num_chars=100):
    print(f"{ch * num_chars}")