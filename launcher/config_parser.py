#!/usr/bin/env python
import base64
import json
import logging
import os
import socket
from pathlib import Path

from savanna.arguments import GlobalConfig


def get_world_info(save_path=None):
    """
    Only needed if using `deepspeed` launcher, doesn't affect `torchrun` if included.
    """
    world_info = {}
    num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
    num_gpus = list(range(num_gpus))
    for host in os.environ["SLURM_JOB_NODELIST"].split("\n"):
        world_info[host] = num_gpus
    world_info_encoded = base64.urlsafe_b64encode(json.dumps(world_info).encode("utf-8")).decode("utf-8")
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(world_info_encoded)
    print(f"World info: {world_info}")
    print(f"World info encoded: {world_info_encoded}")
    return world_info_encoded

def check_slurm_env(hostfile):
    """
    Check SLURM is set up and sets master node.

    Checks:
        - num nodes matches those in hostfile
        - gpus are allocated
        - master port is set

    NOTE: We manually set env var `SLURM_JOB_NODELIST` in order to ensure proper world config
    as `SLURM_JOB_NODELIST` on NVIDIA cluster does not always appear as newline
    separated list of hostnames, which is required for DeepSpeed's internal
    world config.
    """
    with open(hostfile, "r") as f:
        # We do this rather than upfront parsing of hosts since DS relies on SLURM_JOB_NODELIST
        # being a multiline string
        hosts = f.read().strip()

    hostlist = hosts.split("\n")
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", None)

    assert num_nodes is not None, "SLURM_JOB_NUM_NODES not set"
    print(f"SLURM NUM NODES: {num_nodes}")
    
    assert len(hostlist) == int(
        num_nodes
    ), f"Num Hosts (hostfile) != SLURM NNODES: {len(hostlist)} != {num_nodes}\nHostfile:\n{hosts}"

    # Deepspeed expects SLURM_JOB_NODELIST as a list of newline-separated hostnames
    # SLURM_JOB_NODELIST on NVIDIA cluster does not always follow this format
    # Important to set here before global config is set in order to ensure proper world config
    os.environ["SLURM_JOB_NODELIST"] = hosts

    # This should have been done earlier in the script
    master_node = os.environ.get("MASTER_NODE", None)
    assert master_node is not None, "MASTER_NODE is not set"
    assert master_node == hostlist[0], "MASTER_NODE != hostlist[0]"
    master_port = os.environ.get("MASTER_PORT", None)
    assert master_port is not None, "MASTER_PORT is not set"

    num_gpus = os.environ.get("SLURM_GPUS_ON_NODE", None)
    assert num_gpus is not None, "SLURM_GPUS_ON_NODE is not set"
    print(f"SLURM_GPUS_ON_NODE: {num_gpus}")
    
    return master_node, int(num_nodes), int(num_gpus)


# def get_wandb_key(global_config: GlobalConfig):
#     from savanna.utils import get_wandb_api_key

#     wandb_token = get_wandb_api_key(global_config=global_config)
#     return wandb_token


def check_wandb(global_config: GlobalConfig):
    # First check if user exported WANDB_API_KEY
    wandb_token = os.environ.get("WANDB_API_KEY", None)

    if wandb_token is None:
        wandb_credentials_path = Path("~/.netrc").expanduser()
        assert wandb_credentials_path.exists(), f"Wandb credentials not found at {wandb_credentials_path}"
        is_logged_in = False
        for line in wandb_credentials_path.read_text().split("\n"):
            if line.startswith("machine"):
                if line.split(" ")[1].strip() == global_config.wandb_host:
                    is_logged_in = True
                    break

    assert wandb_token is not None or is_logged_in, "Wandb token not found"


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    from savanna.arguments import GlobalConfig

    config_files, hostlist, train_args_output, global_config = GlobalConfig.consume_deepy_args(return_extra_args=True)
    assert len(config_files) == 2
    
    data_config, model_config = config_files
    
    master_node, num_nodes, num_gpus = check_slurm_env(hostlist)
    assert hasattr(global_config, "global_num_gpus")
    assert global_config.global_num_gpus == num_nodes * num_gpus, f"global_num_gpus = {global_config.global_num_gpus} != num_nodes * num_gpus = {num_nodes * num_gpus}"
    
    curr_node = socket.gethostname()

    print(f"MASTER_NODE: {master_node}")

    # Ensure only a single node is generating train args
    if curr_node != master_node:
        print(f"{curr_node} - Not generating train args")
        return
    else:
        print(f"{curr_node} - Generating train args")

        # This part is mostly the same as original `launch.py`
        # Main difference is that the config should have `deepspeed_slurm`
        # set to true, which sets the necessary config vars during
        # deepspeed's internal config parsing
        deepspeed_main_args = global_config.get_deepspeed_main_args()
        
        # These are set externally in the SLURM script
        # assert "--master_port" not in deepspeed_main_args
        # assert "--master_addr" not in deepspeed_main_args
        
        # Save encoded train args to pass to launcher
        # NOTE: downstream parser (`GlobalConfig.consume_global_config` in `train.py`)
        # only parses `deepspeed_config` and `megatron_config` commandline args
        # of which only `megatron_config` is needed, as `deepspeed_config` is
        # only needed for autotuning.
        
        # Parse deepspeed.launcher.launch specific args
        output_dir = Path(train_args_output).parent
        world_info_encoded = get_world_info()
        deepspeed_main_args = ["--world_info", world_info_encoded] + deepspeed_main_args
        print(f"Deepspeed args after world info: {deepspeed_main_args}")
        if global_config.enable_each_rank_log:         
            rank_log_dir = output_dir / "logs"/ Path(model_config).stem / "rank_logs"
            print(f"Setting rank log dir to {rank_log_dir}")
            rank_log_dir.mkdir(parents=True, exist_ok=True)
            deepspeed_main_args = ["--enable_each_rank_log", str(rank_log_dir)] + deepspeed_main_args  
        
        ds_args = " ".join(deepspeed_main_args)
        print(f"Train args: {ds_args}")
        with open(train_args_output, "w") as f:
            f.write(ds_args)

        # Need to check how to set wandb keys in containerized envs
        # Should be propagated if user home is mounted, since wandb key
        # is stored in ~/.netrc assuming user as already logged in.
        if global_config.use_wandb:
            check_wandb(global_config=global_config)
        print(f"{master_node}: Writing train args to {train_args_output}")
        print(f"{master_node}: Done!")

        triton_cache = os.environ.get("TRITON_CACHE_DIR", None)
        if triton_cache is not None:
            print("TRITON_CACHE_DIR: ", triton_cache)
        else:
            print("TRITON_CACHE_DIR NOT FOUND!")
            
if __name__ == "__main__":
    main()
    print("Done!")