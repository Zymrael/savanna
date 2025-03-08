import argparse
from dataclasses import asdict

import torch
from triton.testing import do_bench

from savanna.kernels.triton_src.cgcg.interface import two_pass_chunked_gate_conv_gate
from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    BwdKernelConfigRefactor,
    FwdKernelConfigRefactor,
)
from savanna.kernels.triton_src.cgcg.tests.utils import setup_inputs


def run_bench(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    dtype=torch.bfloat16,
    fwd_autotune=False,
    bwd_autotune=False,
    fwd_kernel_config: FwdKernelConfigRefactor = None,
    bwd_kernel_config: BwdKernelConfigRefactor = None,
):
    assert fwd_autotune or (
        fwd_kernel_config is not None
    ), "Must specify fwd_kernel_cfg if not autotuning fwd"
    assert bwd_autotune or (
        bwd_kernel_config is not None
    ), "Must specify bwd_kernel_cfg if not autotuning bwd"
    dg = d // g

    x, B, C, h = setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True)
    h.retain_grad()

    # Ref grad
    x_ref = x.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    h_ref = h.detach().clone().requires_grad_()

    y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    # Backprop
    dy = torch.randn_like(y_ref)

    def ref_fwd():
        y = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

    def ref_bwd():
        y_ref.backward(dy, retain_graph=True)

    def ref_fwd_bwd():
        y = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)
        y.backward(dy)

    def test_fwd():
        y = two_pass_chunked_gate_conv_gate(
            x,
            B,
            C,
            h,
            bs=bs,
            seqlen=seqlen,
            g=g,
            dg=dg,
            fwd_autotune=fwd_autotune,
            fwd_kernel_cfg=fwd_kernel_config,
            bwd_autotune=bwd_autotune,
            fused_bwd=True,
            bwd_kernel_cfg=bwd_kernel_config,
        )
        return y

    y = test_fwd()

    def test_bwd():
        y.backward(dy, retain_graph=True)

    def test_fwd_bwd():
        y = two_pass_chunked_gate_conv_gate(
            x,
            B,
            C,
            h,
            bs=bs,
            seqlen=seqlen,
            g=g,
            dg=dg,
            fwd_autotune=fwd_autotune,
            fwd_kernel_cfg=fwd_kernel_config,
            bwd_autotune=bwd_autotune,
            fused_bwd=True,
            bwd_kernel_cfg=bwd_kernel_config,
        )
        y.backward(dy)

    # try:
    #     test_fn()
    # except:
    #     print("Kernel config {kernel_config} failed")

    ref_fwd_ms = do_bench(ref_fwd)
    ref_bwd_ms = do_bench(ref_bwd, grad_to_none=[x_ref, B_ref, C_ref, h_ref])
    test_fwd_ms = do_bench(test_fwd)
    test_bwd_ms = do_bench(test_bwd, grad_to_none=[x, B, C, h])

    ref_time_fwdbwd_ms = do_bench(ref_fwd_bwd, grad_to_none=[x_ref, B_ref, C_ref, h_ref])
    test_time_fwdbwd_ms = do_bench(test_fwd_bwd, grad_to_none=[x, B, C, h])

    speedup_fwd = ref_fwd_ms / test_fwd_ms
    speedup_bwd = ref_bwd_ms / test_bwd_ms
    speedup_fwdbwd = ref_time_fwdbwd_ms / test_time_fwdbwd_ms

    print(f"fwd: {ref_fwd_ms} {test_fwd_ms} {speedup_fwd}x")
    print(f"bwd: {ref_bwd_ms} {test_bwd_ms} {speedup_bwd}x")
    print(f"fwdbwd: {ref_time_fwdbwd_ms} {test_time_fwdbwd_ms} {speedup_fwdbwd}x")
    return (
        bs,
        seqlen,
        d,
        g,
        dg,
        filter_size,
        str(dtype),
        asdict(fwd_kernel_config),
        asdict(bwd_kernel_config),
        ref_fwd_ms,
        test_fwd_ms,
        speedup_fwd,
        ref_bwd_ms,
        test_bwd_ms,
        speedup_bwd,
        ref_time_fwdbwd_ms,
        test_time_fwdbwd_ms,
        speedup_fwdbwd,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--g", type=int, default=256)
    parser.add_argument("--filter_size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--fwd_num_stages", type=int, default=1)
    parser.add_argument("--fwd_num_warps", type=int, default=4)
    parser.add_argument("--fwd_chunk_tiles_per_program", type=int, default=1)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    bs = args.bs
    seqlen = args.seqlen
    d = args.d
    g = args.g
    filter_size = args.filter_size

    dg = d // g

    # CHUNK_SIZES = [128, 256]
    # SWIZZLES = ["row"]
    # CHUNK_TILES_PER_PROGRAM = [1, 2, 3]
    # WARPS = [4, 8]
    # STAGES = [1, 2, 3, 4, 5]
    # kernel_configs = list(
    #     itertools.product(CHUNK_SIZES, CHUNK_SIZES, CHUNK_TILES_PER_PROGRAM, SWIZZLES, WARPS, STAGES)
    # )
    # results = []
    print(
        f"Running benchmark with bs={bs}, seqlen={seqlen}, d={d}, g={g}, dg={dg}, filter_size={filter_size}, dtype={dtype}"
    )
    # print(f"Total kernel configs: {len(kernel_configs)}")
    #
    if filter_size == 128:
        fwd_kernel_cfg = FwdKernelConfigRefactor(
            CHUNK_SIZE=128,
            BLOCK_D=min(dg, 128),
            CHUNK_TILES_PER_PROGRAM=1,
            THREADBLOCK_SWIZZLE="row",
            num_warps=4,
            num_stages=5,
        )
    elif filter_size == 7:
        fwd_kernel_cfg = FwdKernelConfigRefactor(
            CHUNK_SIZE=128,
            BLOCK_D=min(dg, 128),
            CHUNK_TILES_PER_PROGRAM=2,
            THREADBLOCK_SWIZZLE="row",
            num_warps=4,
            num_stages=5,
        )
    else:
        raise ValueError
    bwd_kernel_cfg = BwdKernelConfigRefactor(
        pre_conv_BLOCK_X=128,
        pre_conv_BLOCK_Y=128,
        pre_conv_num_warps=8,
        post_conv_BLOCK_X=32,
        post_conv_BLOCK_Y=128,
        post_conv_num_warps=4,
    )
    print(
        f"Running benchmark with fwd kernel config: {fwd_kernel_cfg} and bwd kernel config: {bwd_kernel_cfg}"
    )
    result = run_bench(
        bs,
        seqlen,
        d,
        g,
        filter_size,
        dtype,
        fwd_kernel_config=fwd_kernel_cfg,
        bwd_kernel_config=bwd_kernel_cfg,
    )
