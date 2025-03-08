import itertools
from dataclasses import asdict, dataclass

import torch
from triton.runtime import Config, driver

NUM_SMS = None


def get_grid(BS, M, N, NUM_SMS, BLOCK_M, BLOCK_N):
    num_inner_blocks = (M // BLOCK_M) * (N // BLOCK_N)
    total_blocks = BS * num_inner_blocks
    num_programs = min(NUM_SMS, total_blocks)
    grid = (num_programs, 1, 1)
    # grid_persistent = (132,) #(num_programs-1,)
    return grid


def get_SMS():
    global NUM_SMS
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SMS = properties["multiprocessor_count"]
    return NUM_SMS


@dataclass
class KernelConfig:
    BLOCK_M: int
    BLOCK_N: int
    NUM_PIPELINE_STAGES: int
    # Common triton kernel args
    num_warps: int
    num_ctas: int = 1

    def __str__(self):
        return f"{self.name}({self.BLOCK_M=}, {self.BLOCK_N=}, {self.NUM_PIPELINE_STAGES=}, {self.num_warps=}, {self.num_ctas=})"

    def to_dict(self):
        d = {}
        for k, v in asdict(self).items():
            if k != "name":
                k = "{}_{}".format(self.name, k)
                d[k] = v
        return d


@dataclass
class PreConvKernelConfig(KernelConfig):
    def __post_init__(self):
        self.name = "PRE_CONV_CONFIG"


@dataclass
class PostConvKernelConfig(KernelConfig):
    def __post_init__(self):
        self.name = "POST_CONV_CONFIG"


@dataclass
class ShortHyenaOperatorKernelConfig:
    pre_conv_kernel_config: PreConvKernelConfig
    post_conv_kernel_config: PostConvKernelConfig


# @dataclass(frozen=True)
# class _ShapeConfig:
#     bs: int
#     seqlen: int
#     hn: int
#     np: int
#     hl: int


@dataclass(frozen=True)
class ShapeConfig:
    bs: int
    seqlen: int
    num_groups: int
    d_model: int
    kernel_size: int
    # group_dim: int = field(init=False)

    @property
    def group_dim(self):
        return self.d_model // self.num_groups


def get_debug_autotune_configs():
    block_m = [32]
    block_n = [32]
    num_pipeline_stages = [1]
    num_warps = [4]
    configs = [
        Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_PIPELINE_STAGES": nps, "num_warps": nw}
        )
        for bm, bn, nps, nw in itertools.product(
            block_m, block_n, num_pipeline_stages, num_warps
        )
    ]
    return configs


def get_autotune_configs():
    block_m = [32, 64, 128]
    block_n = [32, 64, 128]
    num_pipeline_stages = [1, 2, 3, 4]
    num_warps = [2, 4, 8]
    configs = [
        Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_PIPELINE_STAGES": nps, "num_warps": nw}
        )
        for bm, bn, nps, nw in itertools.product(
            block_m, block_n, num_pipeline_stages, num_warps
        )
    ]
    return configs


def get_dg_heuristic_fwd():
    def set_block_m(args):
        return min(args["dg"], args["BLOCK_M"])

    return {"BLOCK_M": set_block_m}


def get_dg_heuristic_bwd():
    return {"BLOCK_N": lambda args: min(args["dg"], args["BLOCK_N"])}


def get_autotuned_kernel_configs():
    """
    Returns a list of autotuned kernel configs for the 4 kernels
    used in `hyena_mlp`.

    After kernels have been autotuned, they should have a `best_config` field that is populated
    by the triton autotuner.
    """
    from hyena_mlp.triton.bwd_kernels import (
        _post_conv_bwd_kernel_autotune,
        _pre_conv_bwd_kernel_autotune,
    )
    from hyena_mlp.triton.fwd_kernels import (
        _post_conv_fwd_kernel_autotune,
        _pre_conv_fwd_kernel_autotune,
    )

    pre_conv_fwd_best_config: Config = _pre_conv_fwd_kernel_autotune.best_config
    post_conv_fwd_best_config: Config = _post_conv_fwd_kernel_autotune.best_config
    pre_conv_bwd_best_config: Config = _pre_conv_bwd_kernel_autotune.best_config
    post_conv_bwd_best_config: Config = _post_conv_bwd_kernel_autotune.best_config

    pre_conv_fwd_cfg  = {f"PRE_CONV_FWD_{k}": v for k, v in pre_conv_fwd_best_config.all_kwargs().items()}
    post_conv_fwd_cfg = {f"POST_CONV_FWD_{k}": v for k, v in post_conv_fwd_best_config.all_kwargs().items()}
    pre_conv_bwd_cfg  = {f"PRE_CONV_BWD_{k}": v for k, v in pre_conv_bwd_best_config.all_kwargs().items()}
    post_conv_bwd_cfg = {f"POST_CONV_BWD_{k}": v for k, v in post_conv_bwd_best_config.all_kwargs().items()}
    
    return {
        **pre_conv_fwd_cfg,
        **post_conv_fwd_cfg,
        **pre_conv_bwd_cfg,
        **post_conv_bwd_cfg
    }
