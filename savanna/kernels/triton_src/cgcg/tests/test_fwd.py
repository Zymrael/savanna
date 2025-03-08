import itertools
import os
import sys

import pytest
import torch

from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected, gcg_two_pass_chunked_fwd_corrected
from savanna.kernels.triton_src.cgcg.src._fwd_tma import two_pass_fwd_grouped_tma
from savanna.kernels.triton_src.cgcg.src.fwd import two_pass_fwd_grouped
from savanna.kernels.triton_src.cgcg.utils import correction_toeplitz, toeplitz

from savanna.kernels.triton_src.cgcg.tests.utils import FwdTestResult, set_chunk_size, setup_inputs

torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")

BATCH_SIZES = [2]  # , 4, 8]
SEQLEN = [1024, 8192]
D_SIZES = [4096]
GROUP_SIZES = [1]  # , 4, 8]
FILTER_SIZES = [4, 32, 128]
DTYPES = [torch.float16, torch.float32]
RETURN_TOEPLITZ = [True]
RETURN_Y2 = [True]
SCHEDULE = ["default"]  # , "persistent"]
DEBUG_CONFIGS = [
    (1, 32, 32, 1, 4, torch.float32, True, True, "default"),
    (2, 32, 64, 2, 4, torch.float32, False, False, "default"),
    # (2, 32, 64, 2, 4, torch.float32, False, False, "persistent"),
    (1, 32, 32, 1, 4, torch.float16, True, True, "default"),
    (2, 32, 64, 2, 4, torch.float16, False, False, "default"),
    # (2, 32, 64, 2, 4, torch.float16, False, False, "persistent"),
]
TEST_CONFIGS = list(
    itertools.product(
        BATCH_SIZES,
        SEQLEN,
        D_SIZES,
        GROUP_SIZES,
        FILTER_SIZES,
        DTYPES,
        RETURN_TOEPLITZ,
        RETURN_Y2,
        SCHEDULE,
    )
)


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, dtype, return_toeplitz, return_y2, schedule",
    DEBUG_CONFIGS + TEST_CONFIGS,
    ids=lambda x: str(x),
)
# @pytest.mark.parametrize("USE_PIPELINE", [True]) # NOTE: Pipeline required when not on triton nightly
@pytest.mark.parametrize("USE_TMA", [False])
@pytest.mark.parametrize("version", ["v2"])
@pytest.mark.parametrize("return_bx_lag", [True, False])
def test_two_pass_fwd(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    dtype,
    return_toeplitz,
    return_y2,
    schedule,
    USE_TMA,
    version,
    return_bx_lag,
):
    # if USE_PIPELINE and dtype == torch.float16:
    #     pytest.skip("Pipeline not supported for float16")

    # if filter_size > 64 and not USE_PIPELINE and dtype == torch.float32:
    #     pytest.skip("SMEM not large enough when running without pipeline")
    if USE_TMA:
        pytest.skip("Skip TMA for now")

    dg = d // g
    #    CHUNK_SIZE = min(max(filter_size, 32), seqlen // 2)
    CHUNK_SIZE = set_chunk_size(seqlen, filter_size)

    BLOCK_D = 32 if dg > 32 else dg
    num_warps = 4  # if filter_size < 128 else 2
    num_stages = 4  # if filter_size > 6 else 2
    swizzle = "row"
    autotune = False

    is_interpreter = os.environ.get("TRITON_INTERPRET", "0") == "1"
    if is_interpreter and dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
    else:
        ATOL, RTOL = 1e-2, 1e-2

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype)

    y_ref = gcg_fwd_ref_corrected(x, B, C, h)

    T_ref, T_hat_ref = None, None
    y2_ref = None

    # Only test toeplitz if more than 1 chunk
    if return_toeplitz and seqlen > CHUNK_SIZE:
        h_ = h.flip(-1)[:, 0]
        T_ref = toeplitz(h_, size=CHUNK_SIZE)
        T_hat_ref = correction_toeplitz(h_, size=CHUNK_SIZE)

    if return_y2:
        _, _, _, y2_ref, _ = gcg_two_pass_chunked_fwd_corrected(
            x, B, C, h, gl=CHUNK_SIZE, return_intermediates=True
        )

    kernel_config = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "NUM_PIPELINE_STAGES": 0 if schedule == "default" else 1,
        "num_stages": num_stages,
        "THREADBLOCK_SWIZZLE": swizzle,
    }

    # Test grad
    if USE_TMA:
        y, T, T_hat, y2, bx_lag = two_pass_fwd_grouped_tma(
            x,
            B,
            C,
            h,
            schedule=schedule,
            return_toeplitz=return_toeplitz,
            return_y2=return_y2,
            return_bx_lag=return_bx_lag,
            **kernel_config,
        )
    else:
        # kernel_config.update({"USE_PIPELINE": USE_PIPELINE})
        if version == "v2" and schedule == "persistent":
            pytest.skip("Persistent schedule not supported for v2")
        y, T, T_hat, y2, bx_lag = two_pass_fwd_grouped(
            x,
            B,
            C,
            h,
            version=version,
            schedule=schedule,
            return_toeplitz=return_toeplitz,
            return_y2=return_y2,
            return_bx_lag=return_bx_lag,
            **kernel_config,
        )

    fwd_test_result = FwdTestResult(
        name="FwdTest_FwdPass",
        is_interpreter=is_interpreter,
        schedule=schedule,
        bs=bs,
        seqlen=seqlen,
        d=d,
        g=g,
        dtype=dtype,
        version=version,
        return_toeplitz=return_toeplitz,
        return_y2=True,
        return_bx_lag=return_bx_lag,
        FILTER_LEN=filter_size,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        USE_TMA=USE_TMA,
        y=y,
        y_ref=y_ref,
        T=T,
        T_ref=T_ref,
        T_hat=T_hat,
        T_hat_ref=T_hat_ref,
        y2=y2,
        y2_ref=y2_ref,
        ATOL=ATOL,
        RTOL=RTOL,
    )

    if not fwd_test_result.passed:
        print("\n", fwd_test_result, flush=True, file=sys.stderr)

    assert fwd_test_result.passed
