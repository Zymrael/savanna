from savanna.kernels.triton_src.cgcg.interface import TwoPassChunkedGateConvGate
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    FwdKernelConfig,
)


def test_cgcg_interface():
    L = 1024
    num_heads = 16
    hidden_size_per_partition = 1024
    head_dim = 64

    cgcg_fn = TwoPassChunkedGateConvGate
    cgcg_fwd_config = FwdKernelConfig(
        schedule="default",
        CHUNK_SIZE=128,
        BLOCK_D=128,
        THREADBLOCK_SWIZZLE="row",
        name="fwd",
        version="v2",
        RETURN_TOEPLITZ=True,
        RETURN_Y2=True,
        RETURN_BX_LAG=True,
        num_warps=8,
    )
