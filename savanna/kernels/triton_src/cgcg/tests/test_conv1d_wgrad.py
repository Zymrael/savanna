
import itertools

import pytest
import torch

from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected_noncausal
from savanna.kernels.triton_src.cgcg.src._bwd_two_kernel import cgcg_conv1d_wgrad

pytest.skip("Skip for now", allow_module_level=True)

device = "cuda"
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)

BATCH_SIZES = [1, 2, 4]
SEQLEN = [1024]
D_SIZES = [4096]
GROUP_SIZES = [1, 2, 4]
FILTER_SIZES = [4, 32, 128]
DTYPES = [torch.float32, torch.float16]

TEST_CONFIGS = list(itertools.product(BATCH_SIZES, SEQLEN, D_SIZES, GROUP_SIZES, FILTER_SIZES, DTYPES))


@pytest.mark.parametrize("bs, seqlen, d, g, hl, dtype", TEST_CONFIGS, ids=lambda x: str(x))
def test_conv1d_wgrad(bs, seqlen, hl, d, g, dtype):
    dg = d // g
    ATOL = 1e-4
    RTOL = 1e-4

    x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)
    dy = torch.randn_like(x)

    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    bx_ref, h_grouped_ref, _, y_ref = gcg_fwd_ref_corrected_noncausal(
        x_ref, B_ref, C_ref, h_ref, return_intermediates=True
    )

    y_ref.backward(dy)
    dh_ref = h_ref.grad
    Bx = B * x
    bx, h_grouped, dh = cgcg_conv1d_wgrad(Bx, C, dy, h, return_intermediates=True)

    debug_str = f"{bs=}, {seqlen=}, {d=}, {g=}, {hl=}, {dtype=} "

    if dtype == torch.float16:
        print(bx.dtype, bx_ref.dtype)
    passed = torch.allclose(bx, bx_ref, atol=ATOL, rtol=RTOL)
    if not passed:
        bx_diff = (bx - bx_ref).abs().max()
        print(f"{debug_str}: {bx_diff=}")
        assert passed

    passed = torch.allclose(h_grouped, h_grouped_ref, atol=ATOL, rtol=RTOL)
    if not passed:
        h_group_diff = (h_grouped - h_grouped_ref).abs().max()
        print(f"{debug_str}: {h_group_diff=}")
        assert passed

    passed = torch.allclose(dh, dh_ref, atol=ATOL, rtol=RTOL)
    if not passed:
        dh_diff = (dh - dh_ref).abs().max()
        print(f"{debug_str}: {dh_diff=}")
        assert passed


def quick_bench(bs, seqlen, hl, d, g, dtype):
    from triton.testing import do_bench

    dg = d // g
    x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)
    dy = torch.randn_like(x)
    Bx = B * x

    conv_fn = lambda: cgcg_conv1d_wgrad(Bx, C, dy, h)
    t = do_bench(conv_fn)
    print(f"{t=}")


bs = 1
seqlen = 1024
d = 4096
g = 1
dtype = torch.float16
hl = 128
quick_bench(bs, seqlen, hl, d, g, dtype)
