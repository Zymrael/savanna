import datetime
import os

import torch

from savanna import mpu, print_rank_0
from savanna.arguments import GlobalConfig
from savanna.model import BackbonePipe
from savanna.model_loading import get_shard_by_mp_rank


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_model(global_config, use_cache=False, cast_to_dtype=True):
    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    model = model.to_sequential()

    return model.to(global_config.params_dtype) if cast_to_dtype else model


def print_params(model, msg=""):
    for mname, module in model.named_modules():
        for pname, param in module.named_parameters():
            print_rank_0(f"DEBUG::INFERENCE::params::{msg}::{pname} {param.shape} {param.dtype}", debug=True)

def print_buffers(model, msg=""):
    for mname, module in model.named_modules():
        for bname, buffer in module.named_buffers():
            print_rank_0(f"DEBUG::INFERENCE::buffers::{msg}::{bname} {buffer.shape} {buffer.dtype}", debug=True)


def init_savanna():
    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()
    return global_config

def load_checkpoint(model, global_config):
    # NOTE: @jeromeku: this is not needed since device is set in `_init_distributed`
    # world_size = torch.distributed.get_world_size()
    # local_world_size = torch.cuda.device_count()
    # local_rank = rank % local_world_size
    # torch.cuda.set_device(local_rank)
    # device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    load_rank = rank % global_config.model_parallel_size
    world_size = torch.distributed.get_world_size()
    assert world_size == global_config.model_parallel_size * global_config.context_parallel_size, f"Only tp and cp are supported, {world_size=} != {global_config.model_parallel_size=} * {global_config.context_parallel_size=}"

    #checkpoint_files = get_all_shard_files(global_config.load)
    print_rank_0(f"DEBUG::LOAD_CHECKPOINT rank {rank} load_rank {load_rank}", debug=True)
    checkpoint_dir = global_config.load
    shard_file = get_shard_by_mp_rank(checkpoint_dir, load_rank)

    state_dict = torch.load(shard_file, map_location="cpu")["module"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def check_te_env_vars(global_config):
    rank = torch.distributed.get_rank()
    if rank == 0:
        NCCL_DEBUG = os.environ.get("NCCL_DEBUG", None)
        print(f"NCCL_DEBUG {NCCL_DEBUG}", flush=True)
        dp_group = mpu.get_data_parallel_group()
        print(f"DP Group {torch.distributed.get_process_group_ranks(dp_group)}")
        mp_group = mpu.get_model_parallel_group()
        print(f"MP Group {torch.distributed.get_process_group_ranks(mp_group)}")

        AVOID_RECORD_STREAMS = os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS", None)
        print(f"AVOID_RECORD_STREAMS {AVOID_RECORD_STREAMS}")
        cca_cfg = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)
        print(f"PYTORCH_CUDA_ALLOC_CONFIG {cca_cfg}")

        if global_config.expandable_segments:
            assert cca_cfg is not None
            assert "expandable_segments:True" in cca_cfg

    # These should be exported by the launcher script
    if global_config.use_cp_flash_te:
        assert (
            os.environ["NVTE_FLASH_ATTN"] == "1"
        ), "NVTE_FLASH_ATTN must be enabled when use_cp_flash_te = True"
        assert (
            os.environ["NVTE_FUSED_ATTN"] == "0"
        ), "NVTE_FUSED_ATTN must be disabled when use_cp_flash_te = True"
        assert (
            os.environ["NVTE_UNFUSED_ATTN"] == "0"
        ), "NVTE_UNFUSED_ATTN must be disabled when use_cp_flash_te = True"

    # TransformerEngine Attention Env Vars
    if rank == 0:
        deterministic = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
        nvte_flash = os.environ.get("NVTE_FLASH_ATTN", None)
        nvte_fused = os.environ.get("NVTE_FUSED_ATTN", None)
        nvte_flash_bwd = os.environ.get("NVTE_FUSED_ATTN_USE_FAv2_BWD", None)
        nvte_debug = os.environ.get("NVTE_DEBUG", None)
        nvte_debug_level = os.environ.get("NVTE_DEBUG_LEVEL", None)
        print(f"DEBUG::TRANFORMERENGINE::NVTE_ALLOW_NONDETERMINISTIC_ALGO: {deterministic}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_FLASH_ATTN: {nvte_flash}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_FUSED_ATTN: {nvte_fused}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_FUSED_ATTN_USE_FAv2_BWD: {nvte_flash_bwd}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_DEBUG: {nvte_debug}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_DEBUG_LEVEL: {nvte_debug_level}")

