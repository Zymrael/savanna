import itertools

import pytest
import torch
from pytest_check import check

from savanna.kernels.triton_src.cgcg.interface import two_pass_chunked_gate_conv_gate
from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected
from savanna.kernels.triton_src.cgcg.src.bwd import two_pass_bwd_refactor
from savanna.kernels.triton_src.cgcg.src.bwd_kernels import (
    post_conv_kernel_autotune,
    pre_conv_kernel_autotune,
)
from savanna.kernels.triton_src.cgcg.src.fwd import two_pass_fwd_grouped
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    BwdKernelConfigRefactor,
    FwdKernelConfigRefactor,
)

from .utils import check_numerics_and_shape, setup_inputs, setup_kernel_config

"""
Refactored fwd / bwd

Changes:
- bwd: the backwards pass is now a completely written in torch.
- fwd: no longer need to save toeplitz matrix for bwd
- simplified kernel params: 
    - schedule is always "default"
    - version is always "refactor"
    - always return y2 (output of convolution before applying post-gate)
    - TODO: add fwd autotuning now that bwd no longer dependent on fwd tile sizes 

"""

torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")

BATCH_SIZES = [2]  # , 4, 8]
SEQLEN = [1024, 8192]
D_SIZES = [4096]
GROUP_SIZES = [256]  # , 4, 8]
FILTER_SIZES = [4, 32, 128]
DTYPES = [torch.float32, torch.bfloat16]
RETURN_TOEPLITZ = [False]
RETURN_Y2 = [True]
SCHEDULE = ["default"]
AUTOTUNE = [False, True]

DEBUG_CONFIGS = [
    (1, 32, 32, 1, 4, torch.float32, False),
    (2, 32, 64, 2, 4, torch.float32, False),
    (2, 32, 64, 2, 4, torch.bfloat16, False),  # NOTE: torch.nn.grad.conv1d_{input,weight} fails with float16
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

TESTS_TO_RUN = TEST_CONFIGS


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, dtype, autotune",
    TESTS_TO_RUN,
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("fused", [True, False])
def test_bwd_fused_refactor(bs, seqlen, d, g, filter_size, dtype, autotune, fused):

    dg = d // g

    if dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
        ATOL_dh, RTOL_dh = 1e-3, 1e-3
    else:
        ATOL, RTOL = 1e-3, 1e-3
        ATOL_dh, RTOL_dh = 1e-3, 1e-3

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)

    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    Bx_ref, y2_ref, y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref, return_intermediates=True)

    # Backprop
    dy = 0.1 * torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad
    dB_ref = B_ref.grad
    dC_ref = C_ref.grad
    dh_ref = h_ref.grad

    y2 = y2_ref.contiguous().detach().clone()
    Bx = Bx_ref.contiguous().detach().clone()

    if autotune:
        kernel_config = None
        pre_conv_kernel_autotune.configs = pre_conv_kernel_autotune.configs[:2]
        post_conv_kernel_autotune.configs = post_conv_kernel_autotune.configs[:2]
    else:
        kernel_config = BwdKernelConfigRefactor()

    dx, dB, dC, dh = two_pass_bwd_refactor(
        dy=dy,
        Bx=Bx,
        y2=y2,
        x=x,
        B=B,
        C=C,
        h=h,
        bs=bs,
        seqlen=seqlen,
        g=g,
        dg=dg,
        fused=fused,
        autotune=autotune,
        kernel_config=kernel_config,
    )

    with check:
        if not torch.allclose(dx_ref, dx, atol=ATOL, rtol=RTOL):
            assert check_numerics_and_shape(dx, dx_ref, msg="dx", atol=ATOL, rtol=RTOL, verbose=True)
    with check:
        if not torch.allclose(dB_ref, dB, atol=ATOL, rtol=RTOL):
            assert check_numerics_and_shape(dB, dB_ref, msg="dB", atol=ATOL, rtol=RTOL, verbose=True)
    with check:
        if not torch.allclose(dC_ref, dC, atol=ATOL, rtol=RTOL):
            assert check_numerics_and_shape(dC, dC_ref, msg="dC", atol=ATOL, rtol=RTOL, verbose=True)
    with check:
        if not torch.allclose(dh_ref, dh, atol=ATOL_dh, rtol=RTOL_dh):
            assert check_numerics_and_shape(dh, dh_ref, msg="dh", atol=ATOL_dh, rtol=RTOL_dh, verbose=True)


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, dtype, autotune",
    TESTS_TO_RUN,
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("fused", [True, False])
# @pytest.mark.parametrize("autotune", [False, True])
def test_fwd_bwd_refactor(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    dtype,
    autotune,
    fused,
    schedule="default",
    return_bx=True,
    return_y2=True,
    num_autotune_configs=10,
):
    if seqlen <= 128 and autotune:
        pytest.skip("Skipping autotune for seqlen <= 128")

    dg = d // g

    if dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
        # ATOL_dh, RTOL_dh = 1e-3, 1e-3
    else:
        ATOL, RTOL = 1e-3, 1e-2
        # ATOL_dh, RTOL_dh = 1e-3, 1e-2

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)

    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    Bx_ref, y2_ref, y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref, return_intermediates=True)

    # Backprop
    dy = 0.1 * torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad
    dB_ref = B_ref.grad
    dC_ref = C_ref.grad
    dh_ref = h_ref.grad

    # if autotune:
    #     kernel_config = {}
    #     # Don't need to test all configs, just want to ensure autotuning works
    #     _two_pass_fwd_refactor_autotuned.configs = _two_pass_fwd_refactor_autotuned.configs[
    #         :num_autotune_configs
    #     ]
    # else:
    fwd_kernel_config = setup_kernel_config(seqlen, filter_size=filter_size, dg=dg)

    Bx, y2, y = two_pass_fwd_grouped(
        x,
        B,
        C,
        h,
        schedule=schedule,
        return_bx=return_bx,
        return_y2=return_y2,
        autotune=False,
        **fwd_kernel_config,
    )

    if autotune:
        bwd_kernel_config = None
        pre_conv_kernel_autotune.configs = pre_conv_kernel_autotune.configs[:2]
        post_conv_kernel_autotune.configs = post_conv_kernel_autotune.configs[:2]
    else:
        bwd_kernel_config = BwdKernelConfigRefactor()

    dx, dB, dC, dh = two_pass_bwd_refactor(
        dy=dy,
        Bx=Bx,
        y2=y2,
        x=x,
        B=B,
        C=C,
        h=h,
        bs=bs,
        seqlen=seqlen,
        g=g,
        dg=dg,
        fused=fused,
        autotune=autotune,
        kernel_config=bwd_kernel_config,
    )

    with check:
        # Forward check
        if dtype == torch.float32:
            if not torch.allclose(Bx_ref, Bx, atol=ATOL, rtol=RTOL):
                check_numerics_and_shape(Bx, Bx_ref, msg="Bx", atol=ATOL, rtol=RTOL, verbose=True)
            if not torch.allclose(y2_ref, y2, atol=ATOL, rtol=RTOL):
                check_numerics_and_shape(y2, y2_ref, msg="y2", atol=ATOL, rtol=RTOL, verbose=True)
            if not torch.allclose(y_ref, y, atol=ATOL, rtol=RTOL):
                check_numerics_and_shape(y, y_ref, msg="y", atol=ATOL, rtol=RTOL, verbose=True)

    # Backwards check
    with check:
        if not torch.allclose(dx_ref, dx, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dx, dx_ref, msg="dx", atol=ATOL, rtol=RTOL, verbose=True)

    with check:
        if not torch.allclose(dB_ref, dB, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dB, dB_ref, msg="dB", atol=ATOL, rtol=RTOL, verbose=True)

    with check:
        if not torch.allclose(dC_ref, dC, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dC, dC_ref, msg="dC", atol=ATOL, rtol=RTOL, verbose=True)

    with check:
        if not torch.allclose(dh_ref, dh, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dh, dh_ref, msg="dh", atol=ATOL, rtol=RTOL, verbose=True)


@pytest.mark.parametrize(
    "bs, seqlen, d, g, filter_size, dtype, autotune",
    TESTS_TO_RUN,
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("fused", [True, False])
def test_interface_refactor(
    bs, seqlen, d, g, filter_size, dtype, autotune, fused, schedule="default", num_autotune_configs=10
):
    if seqlen <= 128 and autotune:
        pytest.skip("Skipping autotune for seqlen <= 128")

    dg = d // g

    if dtype == torch.float32:
        ATOL, RTOL = 1e-4, 1e-4
    else:
        ATOL, RTOL = 1e-3, 1e-2

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)
    h.retain_grad()

    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    *_, y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref, return_intermediates=True)

    # Backprop
    dy = 0.1 * torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad
    dB_ref = B_ref.grad
    dC_ref = C_ref.grad
    dh_ref = h_ref.grad

    # if autotune:
    #     kernel_config = {}
    #     fwd_kernel_config = None
    #     # Don't need to test all configs, just want to ensure autotuning works
    #     _two_pass_fwd_refactor_autotuned.configs = _two_pass_fwd_refactor_autotuned.configs[
    #         :num_autotune_configs
    #     ]
    # else:
    kernel_config = setup_kernel_config(seqlen, filter_size=filter_size, dg=dg)
    fwd_kernel_config = FwdKernelConfigRefactor(**kernel_config)

    if autotune:
        bwd_kernel_config = None
        pre_conv_kernel_autotune.configs = pre_conv_kernel_autotune.configs[:2]
        post_conv_kernel_autotune.configs = post_conv_kernel_autotune.configs[:2]
    else:
        bwd_kernel_config = BwdKernelConfigRefactor()

    y = two_pass_chunked_gate_conv_gate(
        x,
        B,
        C,
        h,
        bs=bs,
        seqlen=seqlen,
        g=g,
        dg=dg,
        schedule=schedule,
        fwd_autotune=False,
        fwd_kernel_cfg=fwd_kernel_config,
        fused_bwd=fused,
        bwd_autotune=autotune,
        bwd_kernel_cfg=bwd_kernel_config,
    )

    y.backward(dy)

    dx = x.grad
    dB = B.grad
    dC = C.grad
    dh = h.grad

    with check:
        # Forward check
        if not torch.allclose(y_ref, y, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(y, y_ref, msg="y", atol=ATOL, rtol=RTOL, verbose=True)

    # Backwards check
    with check:
        if not torch.allclose(dx_ref, dx, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dx, dx_ref, msg="dx", atol=ATOL, rtol=RTOL, verbose=True)

    with check:
        if not torch.allclose(dB_ref, dB, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dB, dB_ref, msg="dB", atol=ATOL, rtol=RTOL, verbose=True)
    with check:
        if not torch.allclose(dh_ref, dh, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dh, dh_ref, msg="dh", atol=ATOL, rtol=RTOL, verbose=True)

    with check:
        if not torch.allclose(dC_ref, dC, atol=ATOL, rtol=RTOL):
            check_numerics_and_shape(dC, dC_ref, msg="dC", atol=ATOL, rtol=RTOL, verbose=True)
