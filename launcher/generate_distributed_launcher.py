import sys
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parent
# Root directory of `savanna`
SAVANNA_ROOT = LAUNCHER_DIR.parent
sys.path.insert(0, str(SAVANNA_ROOT))
import json

from launcher.constants import NUM_NSYS_PROFILES
from launcher.search_configs import generate_configs
from launcher.slurm_template import (
    _generate_env_setup,
    _generate_heimdall_cmd,
    _generate_launcher_cmd,
    _generate_parser_cmd,
    _generate_slurm_header,
    _generate_slurm_info,
)
from launcher.utils import (
    check_data_config,
    get_model_configs,
    make_submit_script,
    parse_args,
    prep_paths,
    print_delimiter,
    set_master_port,
    transfer_data_config,
    transfer_model_configs,
    update_config,
    update_model_configs,
)


def generate_slurm_script(model_config: Path, data_config: Path, args, num_configs: int = 1):
    """
    Generate SLURM batch script for launching distributed training jobs with `srun` and `torchrun`.
    """

    # CMDS will be updated in-place each each _generate* function
    CMDS = []

    # Generate #SBATCH header
    _generate_slurm_header(model_config, args, CMDS, num_configs=num_configs)

    # Collect and print SLURM env vars for debugging
    _generate_slurm_info(model_config, args, CMDS)

    # Heimdall - Create the eventual srun output for the otel collector, and symlink back to launcher.
    _generate_heimdall_cmd(model_config, args, CMDS)

    # Set inject config paths and other env vars
    _generate_env_setup(model_config, args, CMDS)

    # Start the actual job
    # 1. Parse config and generate train args in format needed for `GlobalConfig.consume_global_config`
    _generate_parser_cmd(model_config, data_config, args, CMDS)

    # 2. Launcher - `srun` + `torchrun` command for launching the job across and within nodes
    _generate_launcher_cmd(model_config, args, CMDS)

    # Done
    SCRIPT = "".join(CMDS)

    if args.verbose:
        print(f"Generated script:\n{SCRIPT}")

    return SCRIPT


def generate_slurm_scripts(args, model_configs_path, data_config_path):
    # Generate launcher scripts
    output_path = args.output_dir
    scripts_dir = output_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    print(f"SLURM scripts generated in {scripts_dir}")

    num_configs = len(model_configs_path)
    for model_config_path in model_configs_path:
        script = generate_slurm_script(model_config_path, data_config_path, args, num_configs)
        # Write script to output path
        if args.job_array:
            file_name = f"{args.job_name}.sbatch"
        else:
            file_name = f"{model_config_path.stem}.sbatch"
        output_filepath = scripts_dir / file_name
        output_filepath.write_text(script)
        output_filepath.chmod(0o755)

        # only need to generate single array job script
        if args.job_array:
            break

    if num_configs > 1:
        make_submit_script(scripts_dir)


def save_args(args):
    # Copy args to output path
    output_path = args.output_dir
    serialized_args = {k: str(v) for k, v in vars(args).items()}
    args_path = output_path / "args.json"

    with open(args_path, "w") as f:
        json.dump(serialized_args, f, indent=2)
    print(f"Args saved at {args_path}")


def set_nsys_profiling_samples(args):
    if args.nsys and args.profile_every is None:
        if args.num_nodes > 4:
            # Profile only 4 nodes
            # E.g., for 8 nodes, nodes: 0, 4, 8, 12
            profile_every = args.num_nodes // NUM_NSYS_PROFILES
            args.profile_every = profile_every
            print(
                f"Warning: {args.num_nodes} nodes and --nsys-profile-every not set. Setting to {profile_every} to limit to {NUM_NSYS_PROFILES} total profiles.  Set --nsys-profile-every to override."
            )
        else:
            args.profile_every = 1

def main():
    args = parse_args()
    
    model_config_paths = get_model_configs(args)
    
    print_delimiter()
    output_path = prep_paths(args)
    print_delimiter()
    
    if args.search_config is not None:
        assert len(model_config_paths) == 1, "Only single model config supported for search"
        model_config_path = model_config_paths[0]
        model_config = update_config(model_config_path, args)
        search_config = args.search_config

        model_configs_dir, model_configs_path = generate_configs(
            template_yaml=model_config,
            search_config_yaml=search_config,
            output_dir=output_path,
            job_name=args.job_name,
        )
        args.model_configs_dir = model_configs_dir
        print(f"Search configs generated in {model_configs_dir}")
    else:
        model_configs = update_model_configs(model_config_paths, args)
        
        print_delimiter()
        
        model_configs_dir, model_configs_path = transfer_model_configs(
            model_config_paths, model_configs, args
        )

    if args.job_array:
        assert len(model_configs_path) > 1, "Job array requires multiple model configs"

    args.model_configs_dir = model_configs_dir
    check_data_config(args.data_config)
    data_config_path = transfer_data_config(args.data_config, args)
    
    # Update number of nsys profiles
    set_nsys_profiling_samples(args)
    # For distributed launcher
    set_master_port(args)

    # Slurm script generation
    generate_slurm_scripts(args, model_configs_path, data_config_path)

    # Save args to output dir
    save_args(args)
    print_delimiter()


if __name__ == "__main__":
    main()
