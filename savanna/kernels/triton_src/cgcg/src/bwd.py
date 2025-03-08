from typing import Any, Union
import torch
import triton

from savanna.kernels.triton_src.cgcg.src.bwd_kernels import (
    _two_pass_bwd_grouped_kernel_v1,
    _two_pass_bwd_grouped_kernel_v2,
)


def two_pass_bwd_grouped(
    dy: torch.Tensor,
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    y2: torch.Tensor,
    version: str,
    T: torch.Tensor = None,
    T_hat: torch.Tensor = None,
    bx_lag: torch.Tensor = None,
    schedule: str = "default",
    autotune: bool = False,
    CHUNK_SIZE: int = None,
    BLOCK_D: int = None,
    NUM_PIPELINE_STAGES: int = 0,  # for tl.range
    THREADBLOCK_SWIZZLE: str = "row",
    CHUNK_TILES_PER_PROGRAM: int = 1,
    num_warps: int = None,
    # TODO: Make sure to set match these defaults to those in CUDAOptions
    num_stages: int = 3,  # for tl.dot, should default to 3
    num_ctas: int = 1,
    maxnreg: int = None,
    warmup: bool = False,
    return_kernel: bool = False,
    return_wgrad: bool = False,
    return_dgrad: bool = False,
) -> Union[torch.tensor, tuple[triton.compiler.CompiledKernel, tuple[Any], tuple[Any]]]:
    """
    Chunked two-pass backwards kernel with grouped filters

    `g`: number of groups along feature dim:
    `dg`: number of features per group

    Assumptions:
        - g == 1: single filter shared among all features
        - 1 < g < d: `g` groups where each group has `dg` features.  `dg` must be power of 2 and > `16`
        to leverage tensorcores.
        - g == d: each feature has its own filter, not implemented currently since this results in GEMV

    - x, B, C: bs x l x g x dg where g * dg = hidden_dim.
    - h: g x 1 x hl where hl is the filter length and must fit within chunk_size

    Args:
        dy (torch.tensor): (bs, l, g, dg)
        x (torch.tensor): (bs, l, g, dg)
        B (torch.tensor): same shape as x
        C (torch.tensor): same shape as x
        h (torch.tensor): (g, 1, hl)
        y2 (torch.tensor): (bs, l, g, dg) = T_local @ B*x + T_correction @ Bx_lag, saved from forward pass
        autotune (bool): If true, use autotuning.
        schedule (str): One of "default" or "persistent":
        - "default" launches a 1-d grid with num programs == total tiles
        - "persistent" launches num_programs = min(NUM_SM, total_tiles), the idea being that
        reuse of CTAs should allow for better pipelining (hiding memory latency).
        CHUNK_SIZE, BLOCK_D, num_warps, num_stages, NUM_PIPELINE_STAGES: these are for running a manually configured kernel
        If any are specified, all must be specified.
        NOTE: NUM_PIPELINE_STAGES is for pipelining `tl.range` as opposed to `num_stages` which is used for GEMM pipelining.
        warmup (bool): If true, compile the kernel and return the compiled kernel.
        return_kernel (bool): If true, run and return the compiled kernel.
        return_autotune_result (bool): If true, return the autotune result.  Only valid if `autotune=True`.
    Returns:
        Return type dependent on `warmup`, `return_kernel`, `return_autotune_result`
        - default is `dx, dB, dC, dh` the output tensor with shape (bs, l, g, dg)
        - if `warmup=True`, then the compiled kernel (triton.compiler.CompiledKernel) along with kernel args and kernel constexprs are returned
        - if `return_kernel=True`, then the grads are returned along with the kernel (triton.runtime.JITFunction)
        - if `return_autotune_result=True`, then a tuple with the grads and the autotuned result (see AutotunedResult) is returned
    """

    bs, seqlen, g, dg = dy.shape
    filter_shape = h.shape
    hg, _in_channel_div_group, filter_len = filter_shape

    if autotune:
        raise NotImplementedError("Autotuning not implemented for bwd")
    else:
        assert all(
            [
                CHUNK_SIZE,
                BLOCK_D,
                num_warps,
                num_stages is not None,
                NUM_PIPELINE_STAGES is not None,
            ]
        ), "Must specify all of CHUNK_SIZE, BLOCK_D, NUM_PIPELINE_STAGES, num_warps, num_stages, "
        if version == "v1":
            kernel: triton.runtime.JITFunction = _two_pass_bwd_grouped_kernel_v1
        elif version == "v2":
            kernel: triton.runtime.JITFunction = _two_pass_bwd_grouped_kernel_v2
            # kernel: triton.runtime.JITFunction = _two_pass_bwd_grouped_kernel_v2
        elif version == "v3":
            raise NotImplementedError("Skip v3 for now")
            # kernel_dgrad: triton.runtime.JITFunction = (
            #     _two_pass_bwd_grouped_kernel_dgrad
            # )
            # kernel_wgrad: triton.runtime.JITFunction = (
            #     _two_pass_bwd_grouped_kernel_wgrad
            # )
        else:
            raise ValueError(f"version {version} not implemented")

    if CHUNK_SIZE < filter_len:
        raise ValueError("CHUNK_SIZE must be >= filter_len")

    # basic shape checks
    assert dg >= 16, "dg must be >= 8 to use tensor-cores"
    assert x.shape == dy.shape == B.shape == C.shape == y2.shape
    assert hg == g
    assert _in_channel_div_group == 1

    # Add assertion for num_warps, CHUNK_SIZE and hl <= 4
    if filter_len < 128 and seqlen > 1024 and CHUNK_SIZE is not None:
        assert CHUNK_SIZE >= 128, f"{__file__}: CHUNK_SIZE must be >= 128 for hl <= 128 and seqlen > 1024"

    if BLOCK_D is not None:
        assert dg % BLOCK_D == 0, f"{__file__}: dg must be multiple of BLOCK_D"

    # hidden_dim
    d = g * dg

    x = x.reshape(bs, seqlen, d)
    B = B.reshape_as(x)
    C = C.reshape_as(x)
    dy = dy.reshape_as(x)

    # Intermediates from forward pass
    y2 = y2.reshape_as(x)
    batch_stride, row_stride, col_stride = dy.stride()

    if T is not None:
        assert T_hat is not None
        assert (
            T.shape == T_hat.shape == torch.Size([g, CHUNK_SIZE, CHUNK_SIZE])
        ), f"T and T_hat must have same shape, expected {g, CHUNK_SIZE, CHUNK_SIZE}, got {T.shape=}, {T_hat.shape=}"
        assert T.is_contiguous()
        assert T_hat.is_contiguous()
        # Kernel constexpr
        LOAD_T = True
    else:
        LOAD_T = False
    if bx_lag is not None:
        assert bx_lag.shape == torch.Size([bs, seqlen, g, dg])
        assert bx_lag.is_contiguous()
        LOAD_BX_LAG = True
    else:
        LOAD_BX_LAG = False
    # triton kernel pre-condition
    assert dy.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert x.is_contiguous()
    assert y2.is_contiguous()

    # Reshape h to a 2-D tensor
    # TODO: remove?
    h = h.reshape(g, filter_len)
    assert h.is_contiguous()

    # use_autotuner = not any([CHUNK_SIZE, BLOCK_D, num_warps, NUM_PIPELINE_STAGES])
    assert not (
        autotune and warmup
    ), "autotune and warmup are not supported, use return_kernel=True to get the kernel after autotuning"

    if schedule == "default":
        if version == "v2" or version == "v3":

            def _1d_grid(META):
                row_tiles = triton.cdiv(seqlen, META["CHUNK_SIZE"])
                # Each program processes CHUNK_TILES_PER_PROGRAM tiles
                # assert row_tiles % META["CHUNK_TILES_PER_PROGRAM"] == 0
                grid_chunks = triton.cdiv(row_tiles, META["CHUNK_TILES_PER_PROGRAM"])

                col_tiles = triton.cdiv(d, META["BLOCK_D"])
                # total_tiles = bs * row_tiles * col_tiles
                total_programs = bs * grid_chunks * col_tiles

                return (total_programs,)

        else:

            def _1d_grid(META):
                row_tiles = triton.cdiv(seqlen, META["CHUNK_SIZE"])
                col_tiles = triton.cdiv(d, META["BLOCK_D"])
                total_programs = bs * row_tiles * col_tiles

                return (total_programs,)

        NUM_PIPELINE_STAGES = 0
        grid = _1d_grid

    elif schedule == "persistent":
        raise NotImplementedError("Skip persistent schedule for now")
        # grid = lambda META: (
        #     min(
        #         device_props.NUM_SM,
        #         triton.cdiv(seqlen, META["CHUNK_SIZE"])
        #         * triton.cdiv(d, META["BLOCK_D"])
        #         * bs,
        #     ),
        # )
    else:
        raise ValueError(f"schedule {schedule} not implemented")

    dx = torch.zeros_like(x)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)

    num_chunks = triton.cdiv(seqlen, CHUNK_SIZE)
    num_blocks = triton.cdiv(d, BLOCK_D)

    if return_wgrad or version == "v1" or version == "v2":
        dhdT = torch.zeros(
            bs,
            num_chunks,
            num_blocks,
            filter_len,
            CHUNK_SIZE,
            device=x.device,
            dtype=h.dtype,
        )
        dhdTc = torch.zeros_like(dhdT)
        (
            dhdT_batch_stride,
            dhdT_chunk_stride,
            dhdT_block_stride,
            dhdT_row_stride,
            dhdT_col_stride,
        ) = dhdT.stride()

    # if version != "v3":
    kernel_args = (
        dy,
        x,
        B,
        C,
        h,
        # Intermediate activations
        y2,
        T,
        T_hat,
        bx_lag,
        # Outputs
        dx,
        dB,
        dC,
        dhdT,
        dhdTc,
        # Strides
        batch_stride,
        row_stride,
        col_stride,
        dhdT_batch_stride,
        dhdT_chunk_stride,
        dhdT_block_stride,
        dhdT_row_stride,
        dhdT_col_stride,
        # Shapes
        bs,
        seqlen,
        g,
        dg,
    )

    kernel_constexprs = {
        "FILTER_LEN": filter_len,
        "SINGLE_GROUP": g == 1,
        "LOAD_T": LOAD_T,
        "LOAD_BX_LAG": LOAD_BX_LAG,
        "CHUNK_SIZE": CHUNK_SIZE,
        "BLOCK_D": BLOCK_D,
        "THREADBLOCK_SWIZZLE": THREADBLOCK_SWIZZLE,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_ctas": num_ctas,
    }

    if version == "v1":
        kernel_constexprs.update(
            {
                "NUM_PIPELINE_STAGES": NUM_PIPELINE_STAGES,
            }
        )
    elif version == "v2" or version == "v3":
        kernel_constexprs.update({"CHUNK_TILES_PER_PROGRAM": CHUNK_TILES_PER_PROGRAM})
    else:
        raise ValueError(f"version {version} not implemented")

    # Can actually run this with fake tensors (no need for actual kernel tensor args)
    if warmup:
        compiled_kernel: triton.compiler.CompiledKernel = kernel.warmup(
            *kernel_args, **kernel_constexprs, grid=(1,)
        )
        return compiled_kernel, kernel_args, kernel_constexprs
    else:
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](*kernel_args, **kernel_constexprs)
        # else:
        #     if return_dgrad:
        #         compiled_kernel_dgrad: triton.compiler.CompiledKernel = kernel_dgrad[grid](
        #             *kernel_args_dgrad, **kernel_constexprs, LOAD_BX=LOAD_BX
        #         )

        dx = dx.reshape(bs, seqlen, g, dg)
        dB = dB.reshape(bs, seqlen, g, dg)
        dC = dC.reshape(bs, seqlen, g, dg)

        num_blocks_per_filter_group = dg // BLOCK_D

        # if version == "v3":
        #     if return_dgrad:
        #         return dx, dB, dC

        #     if return_wgrad:
        #     # Run filter grad kernel
        #         compiled_kernel_wgrad = kernel_wgrad[grid](
        #             *kernel_args_wgrad, **kernel_constexprs
        #         )
        # Run second reduction pass
        dhdT = dhdT.reshape(bs, num_chunks, g, num_blocks_per_filter_group, filter_len, CHUNK_SIZE)
        dhdTc = dhdTc.reshape_as(dhdT)
        dhdT = dhdT.sum([0, 1, 3, 5]).reshape(*filter_shape)
        dhdTc = dhdTc.sum([0, 1, 3, 5]).reshape_as(dhdT)
        dh = dhdT + dhdTc

    if return_kernel:
        # if version == "v3":
        #     compiled_kernel = compiled_kernel_dgrad, compiled_kernel_wgrad
        return dx, dB, dC, dh, compiled_kernel
    else:
        return dx, dB, dC, dh


# if __name__ == "__main__":
#     from savanna.kernels.triton_src.cgcg.triton.fwd import two_pass_fwd_grouped
#     dtype = torch.float32  # torch.float16 leads to numerical differences
#     torch.set_float32_matmul_precision("highest")
#     device = "cuda"
#     torch.manual_seed(0)

#     # Shapes
#     bs = 2
#     seqlen = 1024
#     hl = 4  # Filter size
#     d = 32
#     g = 1
#     dg = d // g
#     # Only for debugging
#     CHUNK_SIZE = max(hl, 32)  # seqlen // 4
#     BLOCK_D = 32  # if dg > 32 else dg
#     CHUNK_TILES_PER_PROGRAM = 1

#     x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)
#     dy = torch.randn_like(x)

#     y2 = torch.randn_like(x)
#     T = torch.randn(
#         g,
#         CHUNK_SIZE,
#         CHUNK_SIZE,
#         dtype=dtype,
#         device=x.device,
#     )
#     T_hat = torch.randn_like(T)


#     num_warps = 4
#     swizzle = "row"
#     warmup = False
#     autotune = False
#     schedule = "default"
#     return_kernel = False
#     return_toeplitz= False
#     return_autotune_result = False
#     return_bx_lag = True
#     return_y2 = True

#     kernel_config_v1 = {
#         "CHUNK_SIZE": CHUNK_SIZE,
#         "BLOCK_D": BLOCK_D,
#         "NUM_PIPELINE_STAGES": 0,
#         "THREADBLOCK_SWIZZLE": swizzle,
#         "num_warps": num_warps,
#         "num_stages": 2,
#         }
#     y_v1, T, T_hat, y2, bx_lag = two_pass_fwd_grouped(
#         x,
#         B,
#         C,
#         h,
#         version="v1",
#         autotune=autotune,
#         schedule=schedule,
#         warmup=warmup,
#         return_kernel=return_kernel,
#         return_autotune_result=return_autotune_result,
#         return_toeplitz=return_toeplitz,
#         return_y2=return_y2,
#         return_bx_lag=return_bx_lag,
#         **kernel_config_v1,
#     )

#     kernel_config_v1 = {
#         "CHUNK_SIZE": CHUNK_SIZE,
#         "BLOCK_D": BLOCK_D,
#         "NUM_PIPELINE_STAGES": 0,
#         "THREADBLOCK_SWIZZLE": swizzle,
#         "num_warps": num_warps,
#         "num_stages": 2,
#     }
#     kernel_config_v2 = {
#         "CHUNK_SIZE": CHUNK_SIZE,
#         "BLOCK_D": BLOCK_D,
#         "CHUNK_TILES_PER_PROGRAM": CHUNK_TILES_PER_PROGRAM,
#         "THREADBLOCK_SWIZZLE": swizzle,
#         "num_warps": num_warps,
#         "num_stages": 2,
#         "DEBUG": False,
#     }
#     bwd_fn_v1_lag_load = lambda: two_pass_bwd_grouped(
#         dy,
#         x,
#         B,
#         C,
#         h,
#         y2=y2,
#         T=T,
#         T_hat=T_hat,
#         bx_lag=bx_lag,
#         version="v1",
#         schedule=schedule,
#         CHUNK_SIZE=CHUNK_SIZE,
#         BLOCK_D=BLOCK_D,
#         NUM_PIPELINE_STAGES=0,
#         CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
#         num_warps=num_warps,
#     )
#     bwd_fn_v1 = lambda: two_pass_bwd_grouped(
#         dy,
#         x,
#         B,
#         C,
#         h,
#         y2=y2,
#         T=T,
#         T_hat=T_hat,
#         bx_lag=None,
#         version="v1",
#         schedule=schedule,
#         CHUNK_SIZE=CHUNK_SIZE,
#         BLOCK_D=BLOCK_D,
#         NUM_PIPELINE_STAGES=0,
#         CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
#         num_warps=num_warps,
#     )
#     bwd_fn_v2 = lambda: two_pass_bwd_grouped(
#         dy,
#         x,
#         B,
#         C,
#         h,
#         y2,
#         T=T,
#         T_hat=T_hat,
#         bx_lag=bx_lag,
#         version="v2",
#         schedule=schedule,
#         CHUNK_SIZE=CHUNK_SIZE,
#         BLOCK_D=BLOCK_D,
#         NUM_PIPELINE_STAGES=0,
#         CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
#         num_warps=num_warps,
#     )
#     # bwd_fn_v3 = lambda: two_pass_bwd_grouped(
#     #     dy,
#     #     x,
#     #     B,
#     #     C,
#     #     h,
#     #     y2,
#     #     T=T,
#     #     T_hat=T_hat,
#     #     version="v3",
#     #     schedule=schedule,
#     #     CHUNK_SIZE=CHUNK_SIZE,
#     #     BLOCK_D=BLOCK_D,
#     #     NUM_PIPELINE_STAGES=0,
#     #     CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
#     #     num_warps=num_warps,
#     # )
#     dx_v1, dB_v1, dC_v1, dh_v1 = bwd_fn_v1()
#     dx_lag_load_v1, dB_lag_load_v1, dC_lag_load_v1, dh_lag_load_v1 = bwd_fn_v1_lag_load()
#     dx_v2, dB_v2, dC_v2, dh_v2 = bwd_fn_v2()
#     # dx_v2, dB_v2, dC_v2, dh_v2 = bwd_fn_v2()
#     # dx_v3, dB_v3, dC_v3, dh_v3 = bwd_fn_v3()
#     dx_diff = (dx_v1 - dx_lag_load_v1).abs().max()
#     print(f"dx_diff={dx_diff}")
#     dB_diff = (dB_v1 - dB_lag_load_v1).abs().max()
#     print(f"dB_diff={dB_diff}")
#     dC_diff = (dC_v1 - dC_lag_load_v1).abs().max()
#     print(f"dC_diff={dC_diff}")
#     dh_diff = (dh_v1 - dh_lag_load_v1).abs().max()
#     print(f"dh_diff={dh_diff}")
#     dx2_diff = (dx_v1 - dx_v2).abs().max()
#     print(f"dx2_diff={dx2_diff}")
#     dB2_diff = (dB_v1 - dB_v2).abs().max()
#     print(f"dB2_diff={dB2_diff}")
#     dC2_diff = (dC_v1 - dC_v2).abs().max()
#     print(f"dC2_diff={dC2_diff}")
#     dh2_diff = (dh_v1 - dh_v2).abs().max()
#     print(f"dh2_diff={dh2_diff}")

#     # dx_diff_v1v3 = (dx_v1 - dx_v3).abs().max()
#     # dx_diff_v1v2 = (dx_v1 - dx_v2).abs().max()
#     # print(f"dx_diff_v1v3={dx_diff_v1v3}")
#     # print(f"dx_diff_v1v2={dx_diff_v1v2}")
#     # # from triton.testing import do_bench

#     # v1_t = do_bench(bwd_fn_v1)
#     # v2_t = do_bench(bwd_fn_v2)
#     # v3_t = do_bench(bwd_fn_v3)
#     # print(f"v1_t={v1_t}")
#     # print(f"v2_t={v2_t}")
#     # print(f"v3_t={v3_t}")
