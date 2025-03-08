import itertools
import os

import pytest
import torch
from pytest_check import check

from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected
from savanna.kernels.triton_src.cgcg.src.fwd import two_pass_fwd_grouped
from savanna.kernels.triton_src.cgcg.src.fwd_kernels import (
    _two_pass_fwd_refactor_autotuned,
)
from savanna.kernels.triton_src.cgcg.tests.utils import (
    check_numerics_and_shape,
    setup_inputs,
    setup_kernel_config,
)

torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")

BATCH_SIZES = [2]  # , 4, 8]
SEQLEN = [1024, 8192]
D_SIZES = [4096]
GROUP_SIZES = [256]  # , 4, 8]
FILTER_SIZES = [4, 32, 128]
DTYPES = [torch.float32, torch.float16]
RETURN_TOEPLITZ = [False]
RETURN_Y2 = [True]
SCHEDULE = ["default"]
AUTOTUNE = [False]
DEBUG_CONFIGS = [
    (1, 32, 32, 1, 4, torch.float32, False),
    (2, 32, 64, 2, 4, torch.float32, False),
    (2, 32, 64, 2, 4, torch.float16, False),  # NOTE: torch.nn.grad.conv1d_{input,weight} fails with float16
    # (2, 32, 64, 2, 4, torch.bfloat16) #
]
TEST_CONFIGS = list(
    itertools.product(
        BATCH_SIZES,
        SEQLEN,
        D_SIZES,
        GROUP_SIZES,
        FILTER_SIZES,
        DTYPES,
        AUTOTUNE,
    )
)

TESTS_TO_RUN = DEBUG_CONFIGS + TEST_CONFIGS


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, dtype, autotune",
    TESTS_TO_RUN,
    ids=lambda x: str(x),
)
def test_two_pass_fwd_refactor(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    dtype,
    autotune,
    return_bx=True,
    return_y2=True,
    schedule="default",
    num_autotune_configs=50,
):

    dg = d // g

    is_interpreter = os.environ.get("TRITON_INTERPRET", "0") == "1"
    if is_interpreter and dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
    else:
        ATOL, RTOL = 1e-3, 1e-3

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype)
    Bx_ref, y2_ref, y_ref = gcg_fwd_ref_corrected(x, B, C, h, return_intermediates=True)

    if autotune:
        kernel_config = {}
        # Don't need to test all configs, just want to ensure autotuning works
        _two_pass_fwd_refactor_autotuned.configs = _two_pass_fwd_refactor_autotuned.configs[
            :num_autotune_configs
        ]
    else:
        kernel_config = setup_kernel_config(seqlen, filter_size, dg)

    Bx, y2, y = two_pass_fwd_grouped(
        x,
        B,
        C,
        h,
        schedule=schedule,
        return_bx=return_bx,
        return_y2=return_y2,
        autotune=autotune,
        verbose=True,
        **kernel_config,
    )
    with check:
        if not torch.allclose(Bx_ref, Bx, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(Bx_ref, Bx, atol=ATOL, rtol=RTOL, msg="Bx", verbose=True)
    with check:
        if not torch.allclose(y2_ref, y2, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(y2_ref, y2, atol=ATOL, rtol=RTOL, msg="y2", verbose=True)
    with check:
        if not torch.allclose(y_ref, y, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(
                y_ref, y, atol=ATOL, rtol=RTOL, msg="y", verbose=True, should_assert=True
            )
