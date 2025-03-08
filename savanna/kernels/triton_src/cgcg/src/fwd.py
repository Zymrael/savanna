from typing import Any, Union
import torch
import triton

# from savanna.kernels.triton_src.cgcg.triton.fwd_tma import _two_pass_fwd_grouped_tma_kernel
from savanna.kernels.triton_src.cgcg.src.fwd_kernels import (
    _two_pass_fwd_grouped_kernel_v1,
    _two_pass_fwd_grouped_kernel_v2,
)
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    AutotunedResult,
)


def two_pass_fwd_grouped(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    y: torch.Tensor = None,
    return_y2: bool = False,  # y2 = T_local @ B*x + T_hat @ B_lag*x_lag
    return_toeplitz: bool = False,
    return_bx_lag: bool = False,  # Bx_lag = B_lag * x_lag
    verbose: bool = False,
    schedule: str = "default",
    autotune: bool = False,
    version: str = "v1",
    CHUNK_SIZE: int = None,
    BLOCK_D: int = None,
    CHUNK_TILES_PER_PROGRAM: int = 1,
    NUM_PIPELINE_STAGES: int = 0,  # for tl.range
    THREADBLOCK_SWIZZLE: str = "row",
    num_warps: int = 4,
    # TODO: Make sure to set match these defaults to those in CUDAOptions
    num_stages: int = 3,  # for tl.dot, should default to 3
    num_ctas: int = 1,
    maxnreg: int = None,
    warmup: bool = False,
    return_kernel: bool = False,
    return_autotune_result: bool = False,
) -> Union[torch.tensor, tuple[triton.compiler.CompiledKernel, tuple[Any], tuple[Any]]]:
    """
    cgcg 2-pass with grouped filters

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
        x (torch.tensor): (bs, l, g, dg)
        B (torch.tensor): same shape as x
        C (torch.tensor): same shape as x
        h (torch.tensor): (g, 1, hl)
        y (Optional[torch.tensor]): (bs, l, g, dg), pre-allocated output
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
        - default is `y` the output tensor with shape (bs, l, g, dg)
        - if `warmup=True`, then the compiled kernel (triton.compiler.CompiledKernel) along with kernel args and kernel constexprs are returned
        - if `return_kernel=True`, then the `y` is returned along with the kernel (triton.runtime.JITFunction)
        - if `return_autotune_result=True`, then a 2-tuple of `y` and the autotuned result (see AutotunedResult) is returned
    """
    bs, seqlen, g, dg = x.shape

    # basic shape checks
    assert dg >= 16, "dg must be >= 16 to use tensor-cores"
    assert x.shape == B.shape == C.shape
    hg, _in_channel_div_group, filter_len = h.shape
    assert hg == g
    assert _in_channel_div_group == 1

    # hidden_dim
    d = g * dg

    x = x.reshape(bs, seqlen, d)
    B = B.reshape_as(x)
    C = C.reshape_as(x)
    batch_stride, row_stride, col_stride = x.stride()

    # triton kernel pre-condition
    assert x.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()

    # Reshape h to a 2-D tensor
    # TODO: remove?
    h = h.reshape(g, filter_len)
    assert h.is_contiguous()

    # use_autotuner = not any([CHUNK_SIZE, BLOCK_D, num_warps, NUM_PIPELINE_STAGES])
    assert not (
        autotune and warmup
    ), "autotune and warmup are not supported, use return_kernel=True to get the kernel after autotuning"

    if autotune:
        raise NotImplementedError("autotuning not implemented yet")
        # if schedule == "default":
        #     kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_default_autotuned
        # elif schedule == "persistent":
        #     kernel: triton.runtime.JITFunction = (
        #         _two_pass_fwd_grouped_persistent_autotuned
        # )
    else:
        assert all(
            [
                CHUNK_SIZE,
                BLOCK_D,
            ]
        ), "Must specify all of CHUNK_SIZE, BLOCK_D, NUM_PIPELINE_STAGES"

        if version == "v1":
            assert NUM_PIPELINE_STAGES is not None, "Must specify NUM_PIPELINE_STAGES for version v1"
            kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_kernel_v1
        elif version == "v2":
            assert CHUNK_TILES_PER_PROGRAM is not None
            assert schedule == "default", "schedule must be default for version v2"
            kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_kernel_v2
        else:
            raise ValueError(f"Unknown version: {version}")

    if BLOCK_D is not None:
        assert dg % BLOCK_D == 0, f"{__file__}: dg must be multiple of BLOCK_D"

    if filter_len < 128 and seqlen > 1024 and CHUNK_SIZE is not None:
        assert CHUNK_SIZE >= 128, f"{__file__}: CHUNK_SIZE must be >= 128 for hl < 128 and seqlen > 1024"

    if CHUNK_TILES_PER_PROGRAM is not None and CHUNK_SIZE is not None:
        assert triton.cdiv(seqlen, CHUNK_SIZE) % CHUNK_TILES_PER_PROGRAM == 0

    if schedule == "default":
        if version == "v2":

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
        raise NotImplementedError("Skip persistent for now")
        # grid = lambda META: (
        #     min(
        #         NUM_SM,
        #         triton.cdiv(seqlen, META["CHUNK_SIZE"])
        #         * triton.cdiv(d, META["BLOCK_D"])
        #         * bs,
        #     ),
        # )
    else:
        raise ValueError(f"schedule {schedule} not implemented")

    if y is None:
        y = torch.zeros_like(x)

    # For backwards
    if return_y2:
        y2 = torch.empty_like(x)
    else:
        y2 = None

    if return_toeplitz:
        assert CHUNK_SIZE is not None, "CHUNK_SIZE must be specified for return_toeplitz"
        # NOTE: Need to initialize T_hat as zeros, since not all chunks need correction term
        T = torch.zeros(g, CHUNK_SIZE, CHUNK_SIZE, device=x.device, dtype=x.dtype)
        T_hat = torch.zeros_like(T)
    else:
        T = None
        T_hat = None

    if return_bx_lag:
        # NOTE: we only need to allocate (bs, num_chunks - 1, chunk_size, block_d)
        # However, to enable autotuning, we allocate extra chunk_size for bx_lag
        # since we assume that chunk_size is not known beforehand
        bx_lag = torch.zeros_like(x)
    else:
        bx_lag = None

    if verbose:
        print(f"{x.shape=}, {B.shape=}, {C.shape=}, {h.shape=}, {y.shape=}")
        print(f"{bs=} {seqlen=} {g=} {dg=} {filter_len=}")
        print(f"{CHUNK_SIZE=}, {BLOCK_D=}, {num_warps=}, {NUM_PIPELINE_STAGES=}")

    kernel_args = (
        x,
        B,
        C,
        h,
        y,
        T,
        T_hat,
        y2,
        bx_lag,
        batch_stride,
        row_stride,
        col_stride,
        bs,
        seqlen,
        g,
        dg,
    )

    kernel_constexprs = {
        "FILTER_LEN": filter_len,
        "SINGLE_GROUP": g == 1,
        "RETURN_TOEPLITZ": return_toeplitz,
        "RETURN_Y2": return_y2,
        "RETURN_BX_LAG": return_bx_lag,
    }
    if not autotune:
        kernel_constexprs.update(
            {
                "CHUNK_SIZE": CHUNK_SIZE,
                "BLOCK_D": BLOCK_D,
                "THREADBLOCK_SWIZZLE": THREADBLOCK_SWIZZLE,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "num_ctas": num_ctas,
            }
        )
        if version == "v1":
            kernel_constexprs.update(
                {
                    "NUM_PIPELINE_STAGES": NUM_PIPELINE_STAGES,
                }
            )
            # if USE_TMA:
            #     kernel_constexprs.update({"DTYPE": x.dtype})
        else:
            kernel_constexprs.update(
                {
                    "CHUNK_TILES_PER_PROGRAM": CHUNK_TILES_PER_PROGRAM,
                }
            )

    if warmup:
        compiled_kernel: triton.compiler.CompiledKernel = kernel.warmup(
            *kernel_args, **kernel_constexprs, grid=(1,)
        )
        return compiled_kernel, kernel_args, kernel_constexprs
    else:
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](*kernel_args, **kernel_constexprs)

        y = y.reshape(bs, seqlen, g, dg)
        if y2 is not None:
            y2 = y2.reshape(bs, seqlen, g, dg)

        if bx_lag is not None:
            bx_lag = bx_lag.reshape(bs, seqlen, g, dg)

        # If no auxiliary results requested, just return tensors
        if not return_autotune_result:
            if return_kernel:
                return y, T, T_hat, y2, bx_lag, compiled_kernel
            return y, T, T_hat, y2, bx_lag
        else:
            # Autotune path
            keys = [k for k in kernel.cache.keys() if kernel.cache[k] == kernel.best_config]
            # Filter for best key, as best_config can be the same for multiple keys
            # TODO: improve this since this is a bit hacky
            # Key is best key if the kernel args match those of the current kernel args and the dtype is the same
            # Assumption is that dtype is the same for all inputs and output
            best_key = [
                k
                for k in keys
                if k[: len(kernel.key_idx)] == (bs, seqlen, g, dg) and k[len(kernel.key_idx)] == str(x.dtype)
            ]
            assert len(best_key) == 1
            # print(f"Autotune Best Config {kernel.best_config} for keys {best_key}")
            autotune_result = AutotunedResult(best_config=kernel.best_config, key=best_key[0])

            if return_kernel:
                return y, T, T_hat, y2, compiled_kernel, autotune_result

            return y, T, T_hat, y2, autotune_result


# if __name__ == "__main__":
#     from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected
#     from savanna.kernels.triton_src.cgcg.triton._fwd_tma import two_pass_fwd_grouped_tma
#     from savanna.kernels.triton_src.cgcg.triton.kernel_utils import get_kernel_occupancy
#     from savanna.kernels.triton_src.cgcg.utils import correction_toeplitz, toeplitz
#     # Debugging settings -> max precision, set seed
#     dtype = torch.float32  # torch.float16 leads to numerical differences
#     torch.set_float32_matmul_precision("highest")
#     device = "cuda"
#     torch.manual_seed(0)
#     torch.set_printoptions(threshold=10000, precision=1)
#     # Shapes
#     bs = 1
#     seqlen = 64
#     hl = 4  # Filter size
#     d = 32
#     g = 1
#     dg = d // g
#     # Only for debugging
#     CHUNK_SIZE = max(hl, 32)  # seqlen // 4
#     BLOCK_D = 32  # if dg > 32 else dg
#     CHUNK_TILES_PER_PROGRAM = 1
#     x = torch.arange(bs * seqlen * g * dg, device=device, dtype=dtype).reshape(bs, seqlen, g, dg)
#     B = torch.ones(bs, seqlen, g, dg, device=device, dtype=dtype)
#     C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)

#     y_ref = gcg_fwd_ref_corrected(x, B, C, h, interleave=True)
#     Bx = B * x
#     num_chunks = seqlen // CHUNK_SIZE
#     Bx = Bx.reshape(bs, num_chunks, CHUNK_SIZE, d)
#     # We assume that each block stores its lagged chunk in the previous chunk slot
#     # E.g., block 1 loads chunk 0 to calculate bx_lag, then stores in chunk 0 of bx_lag_out
#     # Hence, last chunk of bx_lag will be 0 (assuming we allocate full seqlen chunks)
#     Bx[:,-1,:] = 0
#     Bx_lag_ref = Bx
#     # print(f"x:\n{x}")
#     # print(f"bx_lag_ref:\n{Bx_lag_ref}")
#     Bx_lag_ref = Bx_lag_ref.reshape(bs, seqlen, g, dg)
#     # print(f"bx_lag_ref shape: {Bx_lag_ref.shape}")
#     # h_ = h.flip(-1)
#     # T = toeplitz(h_[:, 0], chunk_size)
#     # T_c = correction_toeplitz(h_[:, 0], chunk_size)

#     # print(f"{target=} {NUM_SM=} {NUM_REGS=} {SIZE_SMEM=} {WARP_SIZE=}")
#     num_warps = 4
#     swizzle = "row"
#     warmup = False
#     autotune = False
#     schedule = "default"
#     return_kernel = False
#     return_autotune_result = False
#     return_toeplitz = False
#     return_y2 = False
#     return_bx_lag = True
#     if warmup:
#         y = torch.empty_like(x)
#     else:
#         y = None

#     if autotune:
#         kernel_config = {}
#     else:
#         #
#         kernel_config_v1 = {
#             "CHUNK_SIZE": CHUNK_SIZE,
#             "BLOCK_D": BLOCK_D,
#             "NUM_PIPELINE_STAGES": 0,
#             "THREADBLOCK_SWIZZLE": swizzle,
#             "num_warps": num_warps,
#             "num_stages": 2,
#         }
#         kernel_config_v2 = {
#             "CHUNK_SIZE": CHUNK_SIZE,
#             "BLOCK_D": BLOCK_D,
#             "CHUNK_TILES_PER_PROGRAM": CHUNK_TILES_PER_PROGRAM,
#             "THREADBLOCK_SWIZZLE": swizzle,
#             "num_warps": num_warps,
#             "num_stages": 2,
#             # "DEBUG": False,
#         }

#     y_v1, T, T_hat, y2, bx_lag = two_pass_fwd_grouped(
#         x,
#         B,
#         C,
#         h,
#         y=y,
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
#     # print(f"bx_lag shape: {bx_lag.shape}")
#     # print(f"bx_lag:\n{bx_lag}")
#     lag_diff = (bx_lag - Bx_lag_ref).abs().max()
#     print(f"{lag_diff=}")
#     y_ref = gcg_fwd_ref_corrected(x, B, C, h, use_causal_conv=False)
#     v1_diff = (y_ref - y_v1).abs().max()
#     # v1_tma_diff = (y_ref - y_v1_tma).abs().max()
#     print(f"{v1_diff=}")
#     # y_v1_tma, *_ = two_pass_fwd_grouped_tma(
#     #     x,
#     #     B,
#     #     C,
#     #     h,
#     #     y=y,
#     #     version="v1",
#     #     autotune=autotune,
#     #     schedule=schedule,
#     #     warmup=warmup,
#     #     return_kernel=return_kernel,
#     #     # return_autotune_result=return_autotune_result,
#     #     return_toeplitz=return_toeplitz,
#     #     return_y2=return_y2,
#     #     **kernel_config_v1,
#     # )
#     y_v2, _, _, _, bx_lag2 = two_pass_fwd_grouped(
#         x,
#         B,
#         C,
#         h,
#         y=y,
#         version="v2",
#         autotune=autotune,
#         schedule=schedule,
#         warmup=warmup,
#         return_kernel=return_kernel,
#         return_autotune_result=return_autotune_result,
#         return_toeplitz=return_toeplitz,
#         return_y2=return_y2,
#         return_bx_lag=return_bx_lag,
#         **kernel_config_v2,
#     )
#     v2_diff = (y_ref - y_v2).abs().max()
#     v2_bx_lag_diff = (bx_lag2 - Bx_lag_ref).abs().max()
#     # ptx = kernel.asm["ptx"]
#     # with open("fwd_v2.ptx", "w") as f:
#     #     f.write(ptx)

#     # print(f"{v1_tma_diff=}")
#     print(f"{v2_diff=}")
#     # if warmup:
#     #     kernel, kernel_args, kernel_constexprs = out
#     #     occ = get_kernel_occupancy(kernel, num_warps)
#     # else:
#     #     if return_kernel:
#     #         if autotune and return_autotune_result:
#     #             y, kernel, autotune_result = out
#     #         else:
#     #             y, kernel = out
#     #     else:
#     #         kernel = None
#     #         y = out
#     # if autotune and return_autotune_result:
#     #     print(f"autotune_result: {autotune_result}")

#     # if warmup and not autotune:
#     #     kernel[(NUM_SM, 1, 1)](*kernel_args)

#     # assert y_ref.shape == y.shape
#     # diff = (y_ref - y).abs().max()
#     # print(f"{diff=}")
#     # print()
