#!/usr/bin/env python

"""
Sanity check that arguments are passed correctly from master node and
comms env is set up correctly.

"""
import sys

import torch

from savanna.arguments import GlobalConfig
from savanna.distributed import check_distributed_vars

DIST_VARS = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"]

if __name__ == "__main__":
    print(f"RECEIVED ARGS: {sys.argv}")
    
    print("BEFORE GLOBAL_CONFIG CONSUME")
    env_vars = check_distributed_vars()
    
    global_config = GlobalConfig.consume_global_config()
    print(f"GLOBAL_CONFIG: {global_config}")

    global_config.configure_distributed_args()

    # These should be set by `torchrun`
    print("AFTER GLOBAL_CONFIG CONSUME")
    env_vars = check_distributed_vars()
    
    assert global_config.global_num_gpus == int(
        env_vars["WORLD_SIZE"]
    ), f"global_num_gpus ({global_config.global_num_gpus}) != WORLD_SIZE ({env_vars['WORLD_SIZE']})"
    
    print("Initializing Torch distributed...")
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"This should write to the rank_logs directory for rank {rank}")
    print(f"{rank} / {world_size - 1}: Torch distributed initialized")
    print(f"This should also write to rank_logs for rank {rank}", file=sys.stderr)