import itertools
import sys

import pytest
import torch

from savanna.kernels.triton_src.cgcg.interface import two_pass_chunked_gate_conv_gate
from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected, gcg_two_pass_chunked_fwd_corrected
from savanna.kernels.triton_src.cgcg.src._bwd_tma import two_pass_bwd_grouped_tma
from savanna.kernels.triton_src.cgcg.src.bwd import two_pass_bwd_grouped
from savanna.kernels.triton_src.cgcg.src.fwd import two_pass_fwd_grouped
from savanna.kernels.triton_src.cgcg.src.kernel_utils import BwdKernelConfig, DeviceProps, FwdKernelConfig
from savanna.kernels.triton_src.cgcg.utils import correction_toeplitz, toeplitz

from .utils import BwdTestResult, FwdTestResult, set_chunk_size, setup_inputs

torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")

# TODO: Add num_ctas > 1 tests

BATCH_SIZES = [1, 2]  # , 4]
SEQLEN = [1024, 8192]
D_SIZES = [4096]
GROUP_SIZES = [1, 2]  # , 4]
FILTER_SIZES = [4, 32, 128]
DTYPES = [torch.float32, torch.float16]
CHUNK_SIZES = [None]
BLOCK_D = [None]
VERSIONS = ["v1", "v2"]
LOAD_TOEPLITZ = [True, False]
SCHEDULE = ["default"]  # , "persistent"]
LOAD_BX_LAG = [True, False]
# DEBUG CONFIGS
COMMON_DEBUG_CONFIGS = [
    # bs, l, d, g, hl, chunk_size, block_d, dtype
    (1, 32, 32, 1, 4, 32, 32 // 1),
    (2, 32, 32, 1, 4, 32, 32 // 1),
    (2, 32, 32, 2, 4, 32, 32 // 2),
    (2, 32, 32, 2, 4, 32 // 2, 32 // 2),
    (2, 32, 64, 2, 4, 32 // 2, 16),
]
DEBUG_CONFIGS = [
    list(cfg) + [dtype, version, load_toeplitz, schedule]
    for dtype, version, load_toeplitz, schedule in itertools.product(
        DTYPES,
        VERSIONS,
        LOAD_TOEPLITZ,
        SCHEDULE,
    )
    for cfg in COMMON_DEBUG_CONFIGS
]

TEST_CONFIGS = list(
    itertools.product(
        BATCH_SIZES,
        SEQLEN,
        D_SIZES,
        GROUP_SIZES,
        FILTER_SIZES,
        CHUNK_SIZES,
        BLOCK_D,
        DTYPES,
        VERSIONS,
        LOAD_TOEPLITZ,
        SCHEDULE,
        # LOAD_BX_LAG
    )
)


TESTS_TO_RUN = DEBUG_CONFIGS + TEST_CONFIGS


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, CHUNK_SIZE, BLOCK_D, dtype, version, load_toeplitz, schedule",
    TESTS_TO_RUN,
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("USE_TMA", [False])
def test_two_pass_bwd(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    CHUNK_SIZE,
    BLOCK_D,
    dtype,
    version,
    load_toeplitz,
    schedule,
    USE_TMA,
    is_interpreter,
    device_props: DeviceProps,
):
    if USE_TMA:
        pytest.skip("Skip TMA for now")
    # if USE_TMA and is_interpreter:
    #     pytest.skip("TMA not supported in interpreter")

    if filter_size > 64 and device_props.SIZE_SMEM < 125000:
        pytest.skip("SMEM not sufficient to run backwards kernel for filter size > 64")

    if CHUNK_SIZE is None:
        CHUNK_SIZE = set_chunk_size(seqlen, filter_size)

    if BLOCK_D is None:
        BLOCK_D = 32

    dg = d // g
    num_warps = 4  # if filter_size < 128 else 2
    num_stages = 2  # 1 if filter_size > 6 else 2
    swizzle = "row"
    autotune = False

    if is_interpreter and dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
        ATOL_dh, RTOL_dh = 1e-2, 1e-3
    else:
        ATOL, RTOL = 1e-2, 1e-3
        ATOL_dh, RTOL_dh = 1e-1, 1e-2

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)

    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    # We need y2 = T_local @ Bx + T_correction @ Bx_lag to calculate dC
    # We can't use the chunked ref for calculating dh since h becomes detached when constructing T_local and T_c
    _, _, _, y2, _ = gcg_two_pass_chunked_fwd_corrected(
        x_ref.detach().clone(),
        B_ref.detach().clone(),
        C_ref.detach().clone(),
        h_ref.detach().clone(),
        gl=CHUNK_SIZE,
        return_intermediates=True,
    )
    if load_toeplitz:
        h_ = h.flip(-1)[:, 0]
        T = toeplitz(h_, CHUNK_SIZE)
        T_hat = correction_toeplitz(h_, CHUNK_SIZE)
    else:
        T = None
        T_hat = None

    y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    # Backprop
    dy = 0.1 * torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad
    dB_ref = B_ref.grad
    dC_ref = C_ref.grad
    dh_ref = h_ref.grad

    kernel_config = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "NUM_PIPELINE_STAGES": 0 if schedule == "default" else 1,
        "CHUNK_TILES_PER_PROGRAM": 1,
        "num_stages": num_stages,
        "THREADBLOCK_SWIZZLE": swizzle,
    }
    if USE_TMA:
        kernel = two_pass_bwd_grouped_tma
    else:
        kernel = two_pass_bwd_grouped

    # Test grad
    dx, dB, dC, dh = kernel(
        dy,
        x,
        B,
        C,
        h,
        y2,
        T=T,
        T_hat=T_hat,
        schedule=schedule,
        autotune=autotune,
        version=version,
        **kernel_config,
    )
    test_result = BwdTestResult(
        name="BwdTest_BwdPass",
        version=version,
        is_interpreter=is_interpreter,
        schedule=schedule,
        bs=bs,
        seqlen=seqlen,
        d=d,
        g=g,
        dtype=dtype,
        load_toeplitz=load_toeplitz,
        load_bx_lag=False,
        FILTER_LEN=filter_size,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        USE_TMA=USE_TMA,
        dx=dx,
        dx_ref=dx_ref,
        dB=dB,
        dB_ref=dB_ref,
        dC=dC,
        dC_ref=dC_ref,
        dh=dh,
        dh_ref=dh_ref,
        ATOL=ATOL,
        RTOL=RTOL,
        ATOL_dh=ATOL_dh,
        RTOL_dh=RTOL_dh,
    )
    if not test_result.passed:
        print("\n", test_result, flush=True, file=sys.stderr)

    assert test_result.passed


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, CHUNK_SIZE, BLOCK_D, dtype, version, load_toeplitz, schedule",
    TESTS_TO_RUN,
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("USE_TMA", [False])
@pytest.mark.parametrize("version_fwd", ["v1", "v2"])
@pytest.mark.parametrize("LOAD_BX_LAG", [True, False])
def test_two_pass_fwd_bwd(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    CHUNK_SIZE,
    BLOCK_D,
    dtype,
    version,
    load_toeplitz,
    schedule,
    USE_TMA,
    version_fwd,
    LOAD_BX_LAG,
    is_interpreter,
    device_props: DeviceProps,
):
    if USE_TMA:
        pytest.skip("Skip TMA for now")
    # if USE_TMA and is_interpreter:
    #     pytest.skip("TMA not supported in interpreter")

    if filter_size > 64 and device_props.SIZE_SMEM < 125000:
        pytest.skip("SMEM not sufficient to run backwards kernel for filter size > 64")

    if seqlen == CHUNK_SIZE:
        pytest.skip("Skip seqlen == CHUNK_SIZE, no correction term")

    if CHUNK_SIZE is None:
        CHUNK_SIZE = set_chunk_size(seqlen, filter_size)

    if BLOCK_D is None:
        BLOCK_D = 32

    dg = d // g
    num_warps = 4  # if filter_size < 128 else 2
    num_stages = 2  # 1 if filter_size > 6 else 2
    swizzle = "row"
    autotune = False

    if is_interpreter and dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
        ATOL_dh, RTOL_dh = 1e-2, 1e-3
    else:
        ATOL, RTOL = 1e-2, 1e-3
        ATOL_dh, RTOL_dh = 1e-1, 1e-2

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)

    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    # Backprop
    dy = 0.1 * torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad
    dB_ref = B_ref.grad
    dC_ref = C_ref.grad
    dh_ref = h_ref.grad

    kernel_config = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "NUM_PIPELINE_STAGES": 0 if schedule == "default" else 1,
        "num_stages": num_stages,
        "THREADBLOCK_SWIZZLE": swizzle,
    }

    y, T, T_hat, y2, bx_lag = two_pass_fwd_grouped(
        x,
        B,
        C,
        h,
        version=version_fwd,
        schedule=schedule,
        return_toeplitz=load_toeplitz,
        return_y2=True,
        return_bx_lag=LOAD_BX_LAG,
        **kernel_config,
    )
    _, _, _, y2_ref, _ = gcg_two_pass_chunked_fwd_corrected(
        x, B, C, h, gl=CHUNK_SIZE, return_intermediates=True
    )

    if load_toeplitz:
        h_ = h.flip(-1)[:, 0]
        T_ref = toeplitz(h_, CHUNK_SIZE)
        T_hat_ref = correction_toeplitz(h_, CHUNK_SIZE)
        assert T.shape == T_ref.shape
        assert T_hat.shape == T_hat_ref.shape
    else:
        T_ref = None
        T_hat_ref = None
        assert T is None and T_hat is None

    # Test fwd
    fwd_test_result = FwdTestResult(
        name="FwdBwdTest_FwdPass",
        version=version_fwd,
        is_interpreter=is_interpreter,
        schedule=schedule,
        bs=bs,
        seqlen=seqlen,
        d=d,
        g=g,
        dtype=dtype,
        return_toeplitz=load_toeplitz,
        return_y2=True,
        return_bx_lag=LOAD_BX_LAG,
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
        # Include extra endline to make it easier to read / filter
        print("\n", fwd_test_result, flush=True, file=sys.stderr)

    # passed = torch.allclose(y, y_ref, rtol=RTOL, atol=ATOL)
    # if not passed:
    #     print_diff(y, y_ref, debug_msg + "y diff: ")

    # Test bwd
    dx, dB, dC, dh = two_pass_bwd_grouped(
        dy,
        x,
        B,
        C,
        h,
        y2=y2,
        T=T,
        T_hat=T_hat,
        bx_lag=bx_lag,
        schedule=schedule,
        autotune=autotune,
        version=version,
        **kernel_config,
    )

    bwd_test_result = BwdTestResult(
        name="FwdBwdTest_BwdPass",
        version=version,
        is_interpreter=is_interpreter,
        schedule=schedule,
        bs=bs,
        seqlen=seqlen,
        d=d,
        g=g,
        dtype=dtype,
        load_toeplitz=load_toeplitz,
        load_bx_lag=LOAD_BX_LAG,
        FILTER_LEN=filter_size,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        USE_TMA=USE_TMA,
        dx=dx,
        dx_ref=dx_ref,
        dB=dB,
        dB_ref=dB_ref,
        dC=dC,
        dC_ref=dC_ref,
        dh=dh,
        dh_ref=dh_ref,
        ATOL=ATOL,
        RTOL=RTOL,
        ATOL_dh=ATOL_dh,
        RTOL_dh=RTOL_dh,
    )
    if not bwd_test_result.passed:
        print("\n", bwd_test_result, flush=True, file=sys.stderr)

    assert fwd_test_result.passed
    assert bwd_test_result.passed


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, CHUNK_SIZE, BLOCK_D, dtype, version, load_toeplitz, schedule",
    DEBUG_CONFIGS + TEST_CONFIGS,
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("USE_TMA", [False, True])
@pytest.mark.parametrize("version_fwd", ["v1", "v2"])
@pytest.mark.parametrize("LOAD_BX_LAG", [False, True])
def test_interface(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    CHUNK_SIZE,
    BLOCK_D,
    dtype,
    version,
    load_toeplitz,
    schedule,
    USE_TMA,
    version_fwd,
    LOAD_BX_LAG,
    is_interpreter,
    device_props: DeviceProps,
):
    if USE_TMA:
        pytest.skip("Skip TMA for now")

    # if USE_TMA and is_interpreter:
    #     pytest.skip("TMA not supported in interpreter")

    if filter_size > 64 and device_props.SIZE_SMEM < 125000:
        pytest.skip("SMEM not sufficient to run backwards kernel for filter size > 64")

    if CHUNK_SIZE is None:
        CHUNK_SIZE = set_chunk_size(seqlen, filter_size)

    if BLOCK_D is None:
        BLOCK_D = 32

    dg = d // g
    num_warps = 4  # if filter_size < 128 else 2
    num_stages = 2  # 1 if filter_size > 6 else 2
    swizzle = "row"
    autotune = False

    if is_interpreter and dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
        ATOL_dh, RTOL_dh = 1e-2, 1e-3
    else:
        ATOL, RTOL = 1e-2, 1e-3
        ATOL_dh, RTOL_dh = 1e-1, 1e-2

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)
    h.retain_grad()
    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()
    # Getting following warning:
    # UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
    # Its .grad attribute won't be populated during autograd.backward().
    # If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor.
    # If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead.
    h_ref.retain_grad()

    fwd_config: FwdKernelConfig = FwdKernelConfig(
        schedule=schedule,
        version=version_fwd,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        THREADBLOCK_SWIZZLE=swizzle,
        NUM_PIPELINE_STAGES=0 if schedule == "default" else 1,
        USE_TMA=USE_TMA,
        RETURN_TOEPLITZ=load_toeplitz,
        RETURN_Y2=True,
        RETURN_BX_LAG=LOAD_BX_LAG,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    bwd_config: BwdKernelConfig = BwdKernelConfig(
        version=version,
        schedule=schedule,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        THREADBLOCK_SWIZZLE=swizzle,
        NUM_PIPELINE_STAGES=0 if schedule == "default" else 1,
        USE_TMA=USE_TMA,
        LOAD_TOEPLITZ=load_toeplitz,
        LOAD_BX_LAG=LOAD_BX_LAG,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    y = two_pass_chunked_gate_conv_gate(
        x,
        B,
        C,
        h,
        return_toeplitz=load_toeplitz,
        schedule=schedule,
        autotune=autotune,
        fwd_kernel_cfg=fwd_config,
        bwd_kernel_cfg=bwd_config,
    )
    # Test forward
    fwd_test_result = FwdTestResult(
        name="InterfaceTest_FwdPass",
        version=version_fwd,
        is_interpreter=is_interpreter,
        schedule=schedule,
        bs=bs,
        seqlen=seqlen,
        d=d,
        g=g,
        dtype=dtype,
        return_toeplitz=load_toeplitz,
        return_y2=True,
        return_bx_lag=LOAD_BX_LAG,
        FILTER_LEN=filter_size,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        USE_TMA=USE_TMA,
        y=y,
        y_ref=y_ref,
        ATOL=ATOL,
        RTOL=RTOL,
    )
    if not fwd_test_result.passed:
        # Include extra endline to make it easier to read / filter
        print("\n", fwd_test_result, flush=True, file=sys.stderr)

    # Test backwards
    dy = 0.1 * torch.randn_like(y_ref)
    y_ref.backward(dy)
    y.backward(dy)

    dx_ref, dB_ref, dC_ref, dh_ref = x_ref.grad, B_ref.grad, C_ref.grad, h_ref.grad
    dx, dB, dC, dh = x.grad, B.grad, C.grad, h.grad

    bwd_test_result = BwdTestResult(
        name="InterfaceTest_BwdPass",
        version=version,
        is_interpreter=is_interpreter,
        schedule=schedule,
        bs=bs,
        seqlen=seqlen,
        d=d,
        g=g,
        dtype=dtype,
        load_toeplitz=load_toeplitz,
        load_bx_lag=LOAD_BX_LAG,
        FILTER_LEN=filter_size,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        USE_TMA=USE_TMA,
        dx=dx,
        dx_ref=dx_ref,
        dB=dB,
        dB_ref=dB_ref,
        dC=dC,
        dC_ref=dC_ref,
        dh=dh,
        dh_ref=dh_ref,
        ATOL=ATOL,
        RTOL=RTOL,
        ATOL_dh=ATOL_dh,
        RTOL_dh=RTOL_dh,
    )
    if not bwd_test_result.passed:
        print("\n", bwd_test_result, flush=True, file=sys.stderr)

    assert fwd_test_result.passed
    assert bwd_test_result.passed
