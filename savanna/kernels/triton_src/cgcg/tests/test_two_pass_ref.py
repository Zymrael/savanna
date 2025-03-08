import itertools
import pytest
import torch

from savanna.kernels.triton_src.cgcg.ref_bwd import gcg_two_pass_chunked_bwd
from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected, gcg_two_pass_chunked_fwd_corrected
from savanna.kernels.triton_src.cgcg.src.bwd_kernels import store_T_kernel, store_Tc_kernel

from .utils import print_diff

torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")

BATCH_SIZES = [1, 2]  # , 4, 8]
GROUP_SIZES = [1, 2, 4]
D_SIZES = [4096]
SEQLEN = [1024]
FILTER_SIZES = [4, 128]
DTYPES = [torch.float32]

TEST_CONFIGS = list(itertools.product(BATCH_SIZES, SEQLEN, D_SIZES, GROUP_SIZES, FILTER_SIZES, DTYPES))


def setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True):
    device = "cuda"
    x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype, requires_grad=requires_grad)
    B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype, requires_grad=requires_grad)
    C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype, requires_grad=requires_grad)
    h = torch.randn(g * filter_size, device=device, dtype=dtype, requires_grad=requires_grad).reshape(
        g, 1, filter_size
    )
    return x, B, C, h


@pytest.mark.parametrize("bs, seqlen, d, g, filter_size, dtype", TEST_CONFIGS)
def test_ref_chunked_fwd(bs, seqlen, d, g, filter_size, dtype):
    # device = "cuda"
    dg = d // g
    chunk_size = seqlen // 2
    # hl = filter_size

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=False)
    y_ref = gcg_fwd_ref_corrected(x, B, C, h)
    y_ref_two_pass = gcg_two_pass_chunked_fwd_corrected(x, B, C, h, gl=chunk_size)

    passed = torch.allclose(y_ref, y_ref_two_pass, atol=1e-4)
    if not passed:
        diff = (y_ref - y_ref_two_pass).abs().max()
        print(f"{bs=} {seqlen=} {d=} {g=} {filter_size=} {dtype=}: {diff=}")
    assert passed


@pytest.mark.parametrize("bs, seqlen, d, g, filter_size, dtype", TEST_CONFIGS)
def test_ref_chunked_bwd(bs, seqlen, d, g, filter_size, dtype):
    dg = d // g
    hl = filter_size
    chunk_size = min(max(filter_size, 32), seqlen // 2)

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
        gl=chunk_size,
        return_intermediates=True,
    )

    y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    # Backprop
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad
    dB_ref = B_ref.grad
    dC_ref = C_ref.grad
    dh_ref = h_ref.grad

    # Test grad
    dx, dB, dC, dh = gcg_two_pass_chunked_bwd(dy, x, y2, B, C, h, gl=chunk_size)

    dx_passed = torch.allclose(dx, dx_ref, atol=1e-4)
    if not dx_passed:
        print_diff(dx, dx_ref, f"{bs=} {seqlen=} {d=} {g=} {filter_size=} {dtype=}, dx diff: ")

    dB_passed = torch.allclose(dB, dB_ref, atol=1e-4)
    if not dB_passed:
        print_diff(dB, dB_ref, f"{bs=} {seqlen=} {d=} {g=} {filter_size=} {dtype=}: dB diff: ")

    dC_passed = torch.allclose(dC, dC_ref, atol=1e-4)
    if not dC_passed:
        print_diff(dC, dC_ref, f"{bs=} {seqlen=} {d=} {g=} {filter_size=} {dtype=}: dC diff: ")

    # NOTE: numerical discrepancy between dh and dh_ref
    dh_passed = torch.allclose(dh, dh_ref, atol=1e-2, rtol=1e-2)
    if not dh_passed:
        print_diff(dh, dh_ref, f"{bs=} {seqlen=} {d=} {g=} {filter_size=} {dtype=}: dh diff: ")

    assert dx_passed and dB_passed and dC_passed and dh_passed


@pytest.mark.parametrize("bs, seqlen, d, g, filter_size, dtype", TEST_CONFIGS)
def test_ref_chunked_bwd_store_reduce(bs, seqlen, d, g, filter_size, dtype):
    dg = d // g
    hl = filter_size
    chunk_size = min(max(filter_size, 32), seqlen // 2)

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
        gl=chunk_size,
        return_intermediates=True,
    )

    y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    # Backprop
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)

    dh_ref = h_ref.grad

    # Test grad
    *_, dT, dTc = gcg_two_pass_chunked_bwd(dy, x, y2, B, C, h, gl=chunk_size, return_dT=True)

    assert dT.shape == torch.Size([g, chunk_size, chunk_size]) and dTc.shape == torch.Size(
        [g, chunk_size, chunk_size]
    )

    # if g > 1:
    #     print(dT.shape, dTc.shape)
    #     pytest.skip("TODO: dTc not implemented for g > 1")

    # dT = dT.reshape(chunk_size, chunk_size)
    # dTc = dTc.reshape(chunk_size, chunk_size)

    #    dhl_buffer, dhc_buffer = torch.zeros(hl, chunk_size, device="cuda", dtype=dtype), torch.zeros(hl, chunk_size, device="cuda", dtype=dtype)
    dhl_buffer, dhc_buffer = torch.zeros(g, hl, chunk_size, device="cuda", dtype=dtype), torch.zeros(
        g, hl, chunk_size, device="cuda", dtype=dtype
    )
    group_stride, row_stride, col_stride = dhl_buffer.stride()
    # Each program converts layout of a grouped filter

    store_T_kernel[(g,)](
        dT,
        dhl_buffer,
        group_stride=group_stride,
        row_stride=row_stride,
        col_stride=col_stride,
        CHUNK_SIZE=chunk_size,
        FILTER_LEN=hl,
    )
    store_Tc_kernel[(g,)](
        dTc,
        dhc_buffer,
        group_stride=group_stride,
        row_stride=row_stride,
        col_stride=col_stride,
        CHUNK_SIZE=chunk_size,
        FILTER_LEN=hl,
    )
    dhl = dhl_buffer.sum(-1).reshape_as(dh_ref)
    dhc = dhc_buffer.sum(-1).reshape_as(dh_ref)
    dh_test = dhc + dhl
    # print(f"dhl:\n{dhl}")
    # print(f"dhc:\n{dhc}")
    # print(f"dh_ref:\n{dh_ref}")
    dh_store_pass = torch.allclose(dh_test, dh_ref, atol=1e-2, rtol=1e-2)
    # print(f"dh_test:\n{dh_test}")
    if not dh_store_pass:
        print_diff(dh_test, dh_ref, f"{bs=} {seqlen=} {d=} {g=} {filter_size=} {dtype=}: dh diff: ")

    assert dh_store_pass
