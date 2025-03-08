import itertools
import math
import warnings
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Union

import numpy as np
import torch
import triton
import triton.language as tl


def torch_dtype_to_triton(dtype):
    dtype_str = str(dtype).split(".")[-1]
    tl_dtype = getattr(tl, dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return tl_dtype


@dataclass
class DeviceProps:
    """
    Singleton instance of DeviceProps.

    Each device has its own singleton instance.
    """

    device: int = torch.cuda.current_device()
    # properties: Dict[any] = field(default_factory=dict, init=False)
    NUM_SM: int = field(init=False)
    NUM_REGS: int = field(init=False)
    SIZE_SMEM: int = field(init=False)
    WARP_SIZE: int = field(init=False)
    target: str = field(init=False)

    _instances = {}

    def __new__(cls, device=None):
        device = device or torch.cuda.current_device()
        if device not in cls._instances:
            instance = super(DeviceProps, cls).__new__(cls)
            cls._instances[device] = instance
        return cls._instances[device]

    def __post_init__(self):
        properties = triton.runtime.driver.active.utils.get_device_properties(self.device)
        self.NUM_SM = properties["multiprocessor_count"]
        self.NUM_REGS = properties["max_num_regs"]
        self.SIZE_SMEM = properties["max_shared_mem"]
        self.WARP_SIZE = properties["warpSize"]
        self.target = triton.runtime.driver.active.get_current_target()


# DEVICE_PROPS = DeviceProps()


@dataclass
class KernelOccupancy:
    """
    Occupancy data for launching a persistent kernel

    Attributes:
        n_regs_used (int): number of registers used by kernel
        n_regs_device (int): number of registers available on device per block
        regs_spilled (int): size of local memory used threads
        (https://github.com/triton-lang/triton/blob/71add52905e561f291737375159bd921eee990d9/third_party/nvidia/backend/driver.c#L125C37-L125C71)
        smem_used (int): size of shared memory used by kernel
        smem_device (int): size of shared memory available on device per block
        register_occupancy (int): register occupancy
        smem_occupancy (int): shared memory occupancy
        occupancy (int): min of register and shared memory occupancy
        num_programs: NUM_SM * occupancy
    """

    n_regs_used: int
    n_regs_device: int
    regs_spills: int
    smem_used: int
    smem_device: int
    register_occupancy: int
    smem_occupancy: int
    occupancy: int
    num_programs: int


def get_kernel_occupancy(kernel: triton.compiler.CompiledKernel, num_warps: int, device_props: DeviceProps):
    """
    Calculates the occupancy of the given kernel and returns the number of
    programs needed for full occupancy for a persistent kernel.

    Args:
        kernel: triton compiler kernel
        num_warps: number of warps to be used in the kernel
        device_props: device properties

    Returns:
        KernelOccupancy
    """
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    regs_per_block = n_regs * device_props.WARP_SIZE * num_warps
    register_occ = occupancy = device_props.NUM_REGS // regs_per_block
    smem_occ = device_props.SIZE_SMEM // size_smem
    occupancy = min(occupancy, device_props.SIZE_SMEM // size_smem)
    num_programs = device_props.NUM_SM * occupancy

    return KernelOccupancy(
        n_regs_used=regs_per_block,
        n_regs_device=device_props.NUM_REGS,
        regs_spills=kernel.n_spills,
        smem_used=size_smem,
        smem_device=device_props.SIZE_SMEM,
        register_occupancy=register_occ,
        smem_occupancy=smem_occ,
        occupancy=occupancy,
        num_programs=num_programs,
    )


@dataclass
class AutotunedResult:
    # cache: Dict[tuple, triton.runtime.Config]
    best_config: triton.runtime.Config
    # cache key for best config
    key: tuple[Any]

    def __repr__(self):
        return f"AutotunedResult(best_config={self.best_config},\nkey={self.key})"


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


@triton.jit
def get_program_ids(pid, tiles_per_seq, d_tiles_per_chunk, chunks_per_seq, SWIZZLE: tl.constexpr = "row"):
    """
    Converts 1-D program id to 3-D grid along batch, chunk (sequence) and d (feature) dimensions.

    Args:
        pid: 1-D program id
        tiles_per_seq: number of tiles along sequence dimension
        d_tiles_per_chunk: number of tiles along d dimension
        chunks_per_seq: number of chunks along sequence dimension
        SWIZZLE: "row" or "col", the threadblock launch order, where axis=0 corresponds to
        sequence dim and axis=1 to feature dimension
            - "row" - row major tile order, where blocks are launched feature dimension then sequence dimension
            - "col" - column major tile order, where blocks are launched sequence dimension then feature dimension
        NOTE: "col" should be more L2-cache friendly when grouping, since the same filter is used for each feature chunk

    """
    if SWIZZLE == "row":
        pid_batch = pid // tiles_per_seq
        pid_d = pid % d_tiles_per_chunk
        pid_chunk = (pid // d_tiles_per_chunk) % chunks_per_seq
    elif SWIZZLE == "col":
        pid_batch = pid // tiles_per_seq
        pid_chunk = pid % chunks_per_seq
        pid_d = (pid // chunks_per_seq) % d_tiles_per_chunk
    else:
        tl.static_assert(False, "Invalid SWIZZLE")
    return pid_batch, pid_d, pid_chunk


# Nightly install of triton launches a separate kernel to flush the TMA cache.
# This can be done within kernel by inserting inline asm (see https://github.com/triton-lang/triton/pull/4342)
# Hence we use the following function which currently in triton main branch instead of those in triton.tools.experimental_descriptor.
def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    TMA_SIZE = 128
    # desc = torch.empty(TMA_SIZE, dtype=torch.int8)
    desc = np.empty(TMA_SIZE, dtype=np.int8)
    # triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
    #     ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc.data_ptr()
    # )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc
    )

    gpu_desc = torch.tensor(desc, device="cuda")

    # TMA cache is not being flushed in between dispacthes, therefore we should
    # manually flush the cache every time we create a new TMA descriptor to make
    # sure the following dispatch don't use stale cache when accessing TMA.
    # flush_TMA_cache[(1, )](gpu_desc, num_warps=1)

    return gpu_desc


# Note: on the first use of a new descriptor, each SM must invalidate the descriptor's
# address in TMA cache via fence.proxy.tensormap::generic.acquire.gpu.
def create_1d_tma_descriptor(ptr, dim, block_dim, element_size):
    TMA_SIZE = 128
    desc = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dim, block_dim, element_size, desc)

    gpu_desc = torch.tensor(desc, device="cuda")

    # TMA cache is not being flushed in between dispacthes, therefore we should
    # manually flush the cache every time we create a new TMA descriptor to make
    # sure the following dispatch don't use stale cache when accessing TMA.
    # flush_TMA_cache[(1, )](gpu_desc, num_warps=1)

    return gpu_desc


@triton.jit
def get_program_order(
    pid,
    num_tiles_batch,
    num_tiles_k,
    chunks_per_batch,
    tiles_per_filter_group,
    THREADBLOCK_SWIZZLE: tl.constexpr = "row",
):
    pid_batch = pid // num_tiles_batch

    if THREADBLOCK_SWIZZLE == "row":
        pid_chunk = (pid // num_tiles_k) % chunks_per_batch
        pid_d = pid % num_tiles_k
    elif THREADBLOCK_SWIZZLE == "col":
        pid_chunk = pid % chunks_per_batch
        pid_d = (pid // chunks_per_batch) % num_tiles_k
    else:
        tl.static_assert(False, "THREADBLOCK_SWIZZLE must be 'row' or 'col'")
    pid_filter_group = pid_d // tiles_per_filter_group

    return pid_batch, pid_chunk, pid_d, pid_filter_group


@dataclass
class FwdKernelResult:
    y: torch.Tensor
    y2: torch.Tensor = None
    T: torch.Tensor = None
    T_hat: torch.Tensor = None
    autotuned_result: AutotunedResult = None
    kernel: triton.compiler.CompiledKernel = None


@dataclass
class BwdKernelResult:
    dx: torch.Tensor
    dB: torch.Tensor
    dC: torch.Tensor
    dh: torch.Tensor
    autotuned_result: AutotunedResult = None
    kernel: triton.compiler.CompiledKernel = None


_KERNEL_CONSTEXPRS = {
    "CHUNK_SIZE",
    "BLOCK_D",
    "THREADBLOCK_SWIZZLE",
    "NUM_PIPELINE_STAGES",
    "CHUNK_TILES_PER_PROGRAM",
    "USE_TMA",
}
_V1_FWD_TUNING_KEYS = _KERNEL_CONSTEXPRS - {"CHUNK_TILES_PER_PROGRAM", "USE_TMA"}
# Backwards requires knowing "CHUNK_SIZE" beforehand
_V1_BWD_TUNING_KEYS = _KERNEL_CONSTEXPRS - {"CHUNK_SIZE", "CHUNK_TILES_PER_PROGRAM", "USE_TMA"}
_V2_FWD_TUNING_KEYS = _KERNEL_CONSTEXPRS - {"NUM_PIPELINE_STAGES", "USE_TMA"}
_V2_BWD_TUNING_KEYS = _KERNEL_CONSTEXPRS - {"CHUNK_SIZE", "NUM_PIPELINE_STAGES", "USE_TMA"}

FWD_KERNEL_CONSTEXPRS = _KERNEL_CONSTEXPRS | {"RETURN_TOEPLITZ", "RETURN_Y2", "RETURN_BX_LAG"}
BWD_KERNEL_CONSTEXPRS = _KERNEL_CONSTEXPRS | {"LOAD_TOEPLITZ", "LOAD_BX_LAG"}
KERNEL_LAUNCH_KWARGS = {"num_warps", "num_stages", "num_ctas"}


@dataclass
class KernelConfig:
    # autotune: bool
    schedule: str
    CHUNK_SIZE: int
    BLOCK_D: int
    THREADBLOCK_SWIZZLE: str
    # TMA Features, sm90+ only
    USE_TMA: bool = False
    # TODO: Add TMA flush
    FLUSH_TMA: bool = False
    # Defaults
    NUM_PIPELINE_STAGES: int = 0
    CHUNK_TILES_PER_PROGRAM: int = 1
    # Common triton kernel kwargs
    num_warps: int = 4
    num_stages: int = 3
    # cta -> threadblock cluster, only relevant for sm90+
    num_ctas: int = 1
    # Leave alone for now
    maxnreg: int = None

    def __post_init__(self):
        assert self.schedule in {"default", "persistent"}
        assert self.THREADBLOCK_SWIZZLE in {"row", "col"}
        assert (
            0 <= self.NUM_PIPELINE_STAGES < 2
        ), "NUM_PIPELINE_STAGES > 1 causes segfault currently, see https://github.com/triton-lang/triton/issues/4368"
        if self.schedule == "default":
            warnings.warn("WARNING: Setting NUM_PIPELINE_STAGES to 0 since schedule is default")
            self.NUM_PIPELINE_STAGES = 0

    def __str__(self):
        fields = ", ".join(
            f"{field.name}={getattr(self, field.name)}" for field in self.__dataclass_fields__.values()
        )
        return f"{self.__class__.__name__}: {fields}"

    def __eq__(self, other):
        if not isinstance(other, KernelConfig):
            return NotImplemented
        return all(getattr(self, f.name) == getattr(other, f.name) for f in fields(self))

    def __hash__(self):
        return hash(tuple(getattr(self, f.name) for f in fields(self)))


@dataclass(eq=False)
class FwdKernelConfig(KernelConfig):
    name: str = "fwd"

    version: str = "v2"
    # Intermediate activations for bwd
    RETURN_TOEPLITZ: bool = True
    RETURN_Y2: bool = True
    RETURN_BX_LAG: bool = True


@dataclass(eq=False)
class BwdKernelConfig(KernelConfig):
    name: str = "bwd"
    version: str = "v2"
    # Whether to load or recompute toeplitz matrices
    LOAD_TOEPLITZ: bool = True
    LOAD_BX_LAG: bool = True


if __name__ == "__main__":
    device_props_1 = DeviceProps(device=0)  # Singleton for device 0
    # device_props_2 = DeviceProps(device=1)  # Singleton for device 1
    device_props_3 = DeviceProps(device=0)  # Should return the same instance as device_props_1

    # Both instances should be the same for device 0
    print(device_props_1 is device_props_3)  # Output: True
    print(device_props_1)
