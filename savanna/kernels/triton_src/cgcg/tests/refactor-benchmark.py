import argparse

# autotune_save_dir = os.path.join(os.getcwd(), "triton_autotune_cache")
# os.environ["TRITON_SAVE_AUTOTUNE"] = "1"
# os.environ["TRITON_SAVE_AUTOTUNE_DIR"] = autotune_save_dir
import itertools
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from triton.testing import do_bench

from savanna.kernels.triton_src.cgcg.interface import two_pass_chunked_gate_conv_gate
from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected
from savanna.kernels.triton_src.cgcg.src.bwd_kernels import (
    post_conv_kernel_autotune,
    pre_conv_kernel_autotune,
)
from savanna.kernels.triton_src.cgcg.src.kernel_utils import FwdKernelConfigRefactor
from savanna.kernels.triton_src.cgcg.tests.utils import setup_inputs


def run_bench(
    bs,
    seqlen,
    d,
    g,
    filter_size,
    dtype=torch.bfloat16,
    autotune=False,
    kernel_config: FwdKernelConfigRefactor = None,
):
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
            x, B, C, h, bs=bs, seqlen=seqlen, g=g, dg=dg, fwd_autotune=autotune, fwd_kernel_cfg=kernel_config, bwd_autotune=True, fused_bwd=True
        )
        return y

    y = test_fwd()

    def test_bwd():
        y.backward(dy, retain_graph=True)

    def test_fwd_bwd():
        y = two_pass_chunked_gate_conv_gate(
            x, B, C, h, bs=bs, seqlen=seqlen, g=g, dg=dg, fwd_autotune=autotune, fwd_kernel_cfg=kernel_config, bwd_autotune=True, fused_bwd=True
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
        asdict(kernel_config),
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--d", type=int, default=8192)
    parser.add_argument("--g", type=int, default=512)
    parser.add_argument("--filter_size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("bench_results"))
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)    
    bs = args.bs
    seqlen = args.seqlen
    d = args.d
    g = args.g
    filter_size = args.filter_size

    dg = d // g
    
    CHUNK_SIZES = [128, 256]
    SWIZZLES = ["row"]
    CHUNK_TILES_PER_PROGRAM = [1, 2, 3]
    WARPS = [4, 8]
    STAGES = [1, 2, 3, 4, 5]
    kernel_configs = list(
        itertools.product(CHUNK_SIZES, CHUNK_SIZES, CHUNK_TILES_PER_PROGRAM, SWIZZLES, WARPS, STAGES)
    )
    results = []
    print(
        f"Running benchmark with bs={bs}, seqlen={seqlen}, d={d}, g={g}, dg={dg}, filter_size={filter_size}, dtype={dtype}"
    )
    print(f"Total kernel configs: {len(kernel_configs)}")

    for (
        chunk_tile_size,
        block_d_tile_size,
        ctiles_per_program,
        swizzle,
        num_warps,
        num_stages,
    ) in kernel_configs:
        kernel_config = FwdKernelConfigRefactor(
            CHUNK_SIZE=chunk_tile_size,
            BLOCK_D=min(dg, block_d_tile_size),
            CHUNK_TILES_PER_PROGRAM=ctiles_per_program,
            THREADBLOCK_SWIZZLE=swizzle,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        print(f"Running benchmark with kernel config: {kernel_config}")
        result = run_bench(bs, seqlen, d, g, filter_size, dtype, kernel_config=kernel_config)
        results.append(result)

    df = pd.DataFrame(
        results,
        columns=[
            "bs",
            "seqlen",
            "d",
            "g",
            "dg",
            "filter_size",
            "dtype",
            "fwd_kernel_config",
            "ref_fwd",
            "test_fwd",
            "speedup_fwd",
            "ref_bwd",
            "test_bwd",
            "speedup_bwd",
            "ref_time_fwdbwd",
            "test_time_fwdbwd",
            "speedup_fwdbwd",
        ],
    )
    ts = datetime.now().strftime("%Y%m%d_%H")
    df = df.sort_values(by="speedup_fwdbwd", ascending=False)
    df["bwd_pre_conv_kernel_config"] = df.apply(lambda _: pre_conv_kernel_autotune.best_config.all_kwargs(), axis=1)
    df["bwd_post_conv_kernel_config"] = df.apply(lambda _: post_conv_kernel_autotune.best_config.all_kwargs(), axis=1)
    output_dir = args.output_dir / ts
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = output_dir / f"two_pass_cgcg_bench_results_{bs=}_{seqlen=}_{d=}_{g=}_{dg=}_{filter_size=}.csv"
    
    df.to_csv(output_path, index=False)
    print(df)    
    
    for key in ["speedup_fwd", "speedup_bwd", "speedup_fwdbwd"]:
        df = df.sort_values(by=key, ascending=False)
        best_config = df.iloc[0]["fwd_kernel_config"]
        best_speedup= df.iloc[0][key]
        print(f"Best {key} speedup: {round(best_speedup, 2)}x with fwd kernel config: {best_config}")
    print(f"Best autotuned bwd pre_conv kernel config: {pre_conv_kernel_autotune.best_config}")
    print(f"Best autotuned bwd post_conv kernel config: {post_conv_kernel_autotune.best_config}") 
    
"""
filter_size = 128, g = 256
Best speedup_fwd speedup: 4.39x with kernel config: {'CHUNK_SIZE': 128, 'BLOCK_D': 16, 'THREADBLOCK_SWIZZLE': 'row', 'schedule': 'default', 'USE_TMA': False, 'FLUSH_TMA': False, 'NUM_PIPELINE_STAGES': 0, 'CHUNK_TILES_PER_PROGRAM': 1, 'num_warps': 4, 'num_stages': 3, 'num_ctas': 1, 'maxnreg': None, 'name': 'fwd', 'version': 'refactor', 'RETURN_BX': True, 'RETURN_Y2': True}
Best speedup_bwd speedup: 1.01x with kernel config: {'CHUNK_SIZE': 128, 'BLOCK_D': 16, 'THREADBLOCK_SWIZZLE': 'row', 'schedule': 'default', 'USE_TMA': False, 'FLUSH_TMA': False, 'NUM_PIPELINE_STAGES': 0, 'CHUNK_TILES_PER_PROGRAM': 3, 'num_warps': 4, 'num_stages': 2, 'num_ctas': 1, 'maxnreg': None, 'name': 'fwd', 'version': 'refactor', 'RETURN_BX': True, 'RETURN_Y2': True}
Best speedup_fwdbwd speedup: 1.17x with kernel config: {'CHUNK_SIZE': 128, 'BLOCK_D': 16, 'THREADBLOCK_SWIZZLE': 'row', 'schedule': 'default', 'USE_TMA': False, 'FLUSH_TMA': False, 'NUM_PIPELINE_STAGES': 0, 'CHUNK_TILES_PER_PROGRAM': 1, 'num_warps': 4, 'num_stages': 2, 'num_ctas': 1, 'maxnreg': None, 'name': 'fwd', 'version': 'refactor', 'RETURN_BX': True, 'RETURN_Y2': True}

filter_size = 7, g = 256
Best speedup_fwd speedup: 2.38x with kernel config: {'CHUNK_SIZE': 128, 'BLOCK_D': 16, 'THREADBLOCK_SWIZZLE': 'row', 'schedule': 'default', 'USE_TMA': False, 'FLUSH_TMA': False, 'NUM_PIPELINE_STAGES': 0, 'CHUNK_TILES_PER_PROGRAM': 1, 'num_warps': 4, 'num_stages': 3, 'num_ctas': 1, 'maxnreg': None, 'name': 'fwd', 'version': 'refactor', 'RETURN_BX': True, 'RETURN_Y2': True}
Best speedup_bwd speedup: 0.98x with kernel config: {'CHUNK_SIZE': 256, 'BLOCK_D': 16, 'THREADBLOCK_SWIZZLE': 'row', 'schedule': 'default', 'USE_TMA': False, 'FLUSH_TMA': False, 'NUM_PIPELINE_STAGES': 0, 'CHUNK_TILES_PER_PROGRAM': 1, 'num_warps': 4, 'num_stages': 2, 'num_ctas': 1, 'maxnreg': None, 'name': 'fwd', 'version': 'refactor', 'RETURN_BX': True, 'RETURN_Y2': True}
Best speedup_fwdbwd speedup: 1.16x with kernel config: {'CHUNK_SIZE': 128, 'BLOCK_D': 16, 'THREADBLOCK_SWIZZLE': 'row', 'schedule': 'default', 'USE_TMA': False, 'FLUSH_TMA': False, 'NUM_PIPELINE_STAGES': 0, 'CHUNK_TILES_PER_PROGRAM': 2, 'num_warps': 4, 'num_stages': 4, 'num_ctas': 1, 'maxnreg': None, 'name': 'fwd', 'version': 'refactor', 'RETURN_BX': True, 'RETURN_Y2': True}
"""