import itertools
import math

import triton
import triton.language as tl

from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    get_program_ids,
    is_power_of_2,
)
from savanna.kernels.triton_src.cgcg.src.toeplitz_kernels import load_correction_toeplitz, load_toeplitz


def get_autotune_configs(
    min_warps=4,
    max_warps=8,
    min_stages=1,
    max_stages=5,
    min_chunk_size=32,
    max_chunk_size=128,
    min_d_block=32,
    max_d_block=128,
    threadblock_swizzle=["row", "col"],
):
    configs = []
    assert is_power_of_2(min_warps) and is_power_of_2(max_warps)
    assert min_chunk_size % 16 == 0 and max_chunk_size % 16 == 0
    warp_range = [2**i for i in range(int(math.log2(min_warps)), int(math.log2(max_warps) + 1))]
    stage_range = list(range(min_stages, max_stages + 1))
    chunk_range = [2**i for i in range(int(math.log2(min_chunk_size)), int(math.log2(max_chunk_size) + 1))]
    d_block_range = [2**i for i in range(int(math.log2(min_d_block)), int(math.log2(max_d_block) + 1))]
    configs = [
        triton.Config(
            {
                "CHUNK_SIZE": c,
                "BLOCK_D": d,
                "NUM_PIPELINE_STAGES": 0,
                "THREADBLOCK_SWIZZLE": swz,
            },
            num_warps=w,
            num_stages=stg,
        )
        for w, stg, c, d, swz in itertools.product(
            warp_range, stage_range, chunk_range, d_block_range, threadblock_swizzle
        )
    ]

    # TODO: Set num_stages = 0?
    # for cfg in configs:
    #     cfg.num_stages = 0
    return configs


def get_persistent_configs(
    min_pipeline_stages=0,
    max_pipeline_stages=1,
    min_warps=4,
    max_warps=8,
    min_stages=1,
    max_stages=5,
    min_chunk_size=32,
    max_chunk_size=128,
    min_d_block=32,
    max_d_block=128,
):
    pipeline_range = range(min_pipeline_stages, max_pipeline_stages + 1)
    configs = get_autotune_configs(
        min_warps=min_warps,
        max_warps=max_warps,
        min_stages=min_stages,
        max_stages=max_stages,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        min_d_block=min_d_block,
        max_d_block=max_d_block,
    )
    for cfg in configs:
        for p in pipeline_range:
            cfg.kwargs.update({"NUM_PIPELINE_STAGES": p})
    return configs


# @triton.jit
# def toeplitz_idx(
#     FILTER_LEN: tl.constexpr,
#     CHUNK_SIZE: tl.constexpr,
#     TOEPLITZ_TYPE: tl.constexpr = "toeplitz",
# ):
#     """
#     Generates pointer indices relative to a base filter pointer for materializing toeplitz / correction toeplitz matrix
#     directly on-chip.
#     """
#     if TOEPLITZ_TYPE == "toeplitz":
#         r_idx = tl.arange((FILTER_LEN - 1), CHUNK_SIZE + (FILTER_LEN - 1))[None, :]
#     elif TOEPLITZ_TYPE == "correction_toeplitz":
#         r_idx = (
#             tl.arange((FILTER_LEN - 1), CHUNK_SIZE + (FILTER_LEN - 1))[None, :]
#             - CHUNK_SIZE
#         )
#     else:
#         tl.static_assert(False, "Invalid ToeplitzType")
#     c_idx = tl.arange(0, CHUNK_SIZE)[:, None]
#     idx = r_idx - c_idx
#     return idx


# @triton.jit
# def load_toeplitz(
#     h_ptr,
#     FILTER_LEN: tl.constexpr,
#     CHUNK_SIZE: tl.constexpr,
#     SINGLE_GROUP: tl.constexpr = True,
#     group_num=0,
# ):
#     t_idx = toeplitz_idx(FILTER_LEN, CHUNK_SIZE, "toeplitz")
#     mask = (0 <= t_idx) & (t_idx < FILTER_LEN)

#     if SINGLE_GROUP:
#         T = tl.load(
#             h_ptr + t_idx, mask=mask, other=0.0, eviction_policy="evict_last"
#         )  # Want T to stay resident in L2 cache
#     else:
#         T = tl.load(
#             h_ptr + group_num * FILTER_LEN + t_idx,
#             mask=mask,
#             other=0.0,
#             eviction_policy="evict_last",
#         )

#     return T


# @triton.jit
# def load_correction_toeplitz(
#     h_ptr,
#     FILTER_LEN: tl.constexpr,
#     CHUNK_SIZE: tl.constexpr,
#     SINGLE_GROUP: tl.constexpr = True,
#     group_num=0,
# ):
#     t_idx = toeplitz_idx(FILTER_LEN, CHUNK_SIZE, "correction_toeplitz")
#     mask = (0 <= t_idx) & (t_idx < FILTER_LEN)

#     if SINGLE_GROUP:
#         T_C = tl.load(
#             h_ptr + t_idx, mask=mask, other=0.0, eviction_policy="evict_last"
#         )  # Want T to stay resident in L2 cache
#     else:
#         T_C = tl.load(
#             h_ptr + group_num * FILTER_LEN + t_idx,
#             mask=mask,
#             other=0.0,
#             eviction_policy="evict_last",
#         )

#     return T_C


# TODO:
# Compiler
# - Autotune
# - cache hints
# - fp8 fast accum
# Persistent kernel launch config
#  - Figure out why num_pipeline_stages > 1 causes segfault
#  - Increase number of programs according to occupancy
#  - Improve threadblock swizzling (row-major -> col-major)
# Fusion
# - Single fused kernel (current)
# - 2 separate kernels: y = B * x @ T_local then z = C * (y + B_lag * x @ T_correction)


def get_two_pass_heuristics(include_grouped_heuristic=False, include_dg_check=True):
    # Is this necessary? Assumption is that filter_len needs to be <= chunk_size
    chunk_size_heuristic = lambda args: max(args["FILTER_LEN"], args["CHUNK_SIZE"])
    single_group_heuristic = lambda args: args["g"] == 1
    # Each CTA processes a single filter group
    dg_heuristic = lambda args: min(args["dg"], args["BLOCK_D"])

    heuristics = {"CHUNK_SIZE": chunk_size_heuristic}
    if include_grouped_heuristic:
        heuristics["SINGLE_GROUP"] = single_group_heuristic
    if include_dg_check:
        heuristics["BLOCK_D"] = dg_heuristic
    return triton.heuristics(heuristics)


@triton.jit
def _two_pass_fwd_grouped_kernel_v1(
    # Pointers
    x_ptr,
    B_ptr,
    C_ptr,
    h_ptr,
    out_ptr,
    T_ptr,
    T_hat_ptr,
    y2_ptr,
    bx_lag_ptr,
    # Strides
    batch_stride,
    row_stride,
    col_stride,
    # Shapes
    bs,
    seqlen,
    g,
    dg,
    # Compile-time constant
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    SINGLE_GROUP: tl.constexpr,
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
    # Common triton kernel params
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    DEBUG: tl.constexpr = False,
    # Bwd
    RETURN_TOEPLITZ: tl.constexpr = False,
    RETURN_Y2: tl.constexpr = False,
    RETURN_BX_LAG: tl.constexpr = False,
):
    if DEBUG:
        if tl.program_id(0) == 0:
            tl.static_print(
                "TWO_PASS CONSTEXPRS:\n",
                "FILTER_LEN:",
                FILTER_LEN,
                "CHUNK_SIZE:",
                CHUNK_SIZE,
                "BLOCK_D:",
                BLOCK_D,
                "SINGLE_GROUP:",
                SINGLE_GROUP,
                "THREADBLOCK_SWIZZLE:",
                THREADBLOCK_SWIZZLE,
            )
    # TODO: move this to heuristic
    # tl.device_assert(dg % BLOCK_D == 0, "dg must be multiple of BLOCK_D")

    # Map 1D grid to 3D logical coordinates
    hidden_dim = g * dg
    chunks_per_seq = tl.cdiv(seqlen, CHUNK_SIZE)
    d_tiles_per_chunk = tl.cdiv(hidden_dim, BLOCK_D)
    tiles_per_seq = chunks_per_seq * d_tiles_per_chunk
    chunk_stride = CHUNK_SIZE * hidden_dim
    batch_stride = chunk_stride * chunks_per_seq
    total_tiles = bs * tiles_per_seq

    # Grid stride
    start_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    row_range = tl.arange(0, CHUNK_SIZE)[:, None] * row_stride
    col_range = (
        tl.arange(0, BLOCK_D)[None, :] * col_stride
    )  # not needed, since should be contiguous along feature dim

    for tile_id in tl.range(start_pid, total_tiles, num_programs, num_stages=NUM_PIPELINE_STAGES):
        pid_batch, pid_d, pid_chunk = get_program_ids(
            tile_id, tiles_per_seq, d_tiles_per_chunk, chunks_per_seq
        )

        # First determine offset by batch
        batch_offset = pid_batch * batch_stride
        # Next determine offset by chunk
        chunk_offset = pid_chunk * chunk_stride
        # Next determine offset along feature dim (d)
        col_offset = pid_d * BLOCK_D
        # Map col_offset to filter group
        filter_group = col_offset // dg

        offsets = batch_offset + chunk_offset + col_offset + row_range + col_range

        x = tl.load(x_ptr + offsets)
        B = tl.load(B_ptr + offsets)
        C = tl.load(C_ptr + offsets)

        T = load_toeplitz(
            h_ptr,
            FILTER_LEN,
            CHUNK_SIZE,
            SINGLE_GROUP=SINGLE_GROUP,
            group_num=filter_group,
        )

        Bx = B * x

        y = tl.dot(
            T,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=out_dtype,
        )

        if RETURN_TOEPLITZ:
            blocks_per_filter_group = dg // BLOCK_D
            is_first_pid_in_group = (pid_d % blocks_per_filter_group) == 0
            t_group_stride = CHUNK_SIZE * CHUNK_SIZE
            t_offsets = (
                filter_group * t_group_stride
                + tl.arange(0, CHUNK_SIZE)[:, None] * CHUNK_SIZE
                + tl.arange(0, CHUNK_SIZE)[None, :]
            )
            # Only need to store once per batch per seq per filter group
            if pid_batch == 0 and pid_chunk == 0:
                # Only first program for each filter group needs to store
                if is_first_pid_in_group:
                    tl.store(T_ptr + t_offsets, T)

        # Now load B_lag and x_lag
        # First chunk (t = 0) does not need correction
        if pid_chunk > 0:
            T_c = load_correction_toeplitz(
                h_ptr,
                FILTER_LEN,
                CHUNK_SIZE,
                SINGLE_GROUP=SINGLE_GROUP,
                group_num=filter_group,
            )
            lag_offsets = offsets - chunk_stride
            B_lag = tl.load(B_ptr + lag_offsets)
            x_lag = tl.load(x_ptr + lag_offsets)
            Bx_lag = B_lag * x_lag

            if RETURN_BX_LAG:
                tl.store(bx_lag_ptr + lag_offsets, Bx_lag)

            correction_term = tl.dot(
                T_c,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=out_dtype,
            )

            y += correction_term

            if RETURN_TOEPLITZ:
                # First chunk doesn't calculate correction toeplitz
                # pid_batch = 0, pid_chunk = 1 is the first batch / chunk that calculates correction term
                if pid_batch == 0 and pid_chunk == 1:
                    if is_first_pid_in_group:
                        tl.store(T_hat_ptr + t_offsets, T_c)

        if RETURN_Y2:
            tl.store(y2_ptr + offsets, y)

        y *= C
        out_idx = out_ptr + offsets
        tl.store(out_idx, y)


@triton.jit
def _two_pass_fwd_grouped_kernel_v2(
    # Pointers
    x_ptr,
    B_ptr,
    C_ptr,
    h_ptr,
    out_ptr,
    T_ptr,
    T_hat_ptr,
    y2_ptr,
    bx_lag_ptr,
    # Strides
    batch_stride,
    row_stride,
    col_stride,
    # Shapes
    bs,
    seqlen,
    g,
    dg,
    # Compile-time constant
    FILTER_LEN: tl.constexpr,
    # Autotuned params
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    # Set by heuristic
    SINGLE_GROUP: tl.constexpr,
    CHUNK_TILES_PER_PROGRAM: tl.constexpr = 1,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
    ENABLE_CHECK: tl.constexpr = False,
    # Intermediates for Bwd
    RETURN_TOEPLITZ: tl.constexpr = False,
    RETURN_Y2: tl.constexpr = False,
    RETURN_BX_LAG: tl.constexpr = False,
    DEBUG: tl.constexpr = False,
    # Common triton kernel params
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
):
    if DEBUG:
        if tl.program_id(0) == 0:
            tl.static_print(
                "TWO_PASS CONSTEXPRS:\n",
                "FILTER_LEN:",
                FILTER_LEN,
                "CHUNK_SIZE:",
                CHUNK_SIZE,
                "BLOCK_D:",
                BLOCK_D,
                "SINGLE_GROUP:",
                SINGLE_GROUP,
                "THREADBLOCK_SWIZZLE:",
                THREADBLOCK_SWIZZLE,
                "CHUNK_TILES_PER_PROGRAM:",
                CHUNK_TILES_PER_PROGRAM,
            )
    # TODO: move this to heuristic
    # tl.device_assert(dg % BLOCK_D == 0, "dg must be multiple of BLOCK_D")

    # Map 1D grid to 3D logical coordinates
    hidden_dim = g * dg
    chunks_per_seq = tl.cdiv(seqlen, CHUNK_SIZE)
    effective_chunks_per_seq = chunks_per_seq // CHUNK_TILES_PER_PROGRAM
    d_tiles_per_chunk = tl.cdiv(hidden_dim, BLOCK_D)
    tiles_per_seq = chunks_per_seq * d_tiles_per_chunk
    chunk_stride = CHUNK_SIZE * hidden_dim
    batch_stride = chunk_stride * chunks_per_seq

    # Grid stride
    start_pid = tl.program_id(0)

    row_range = tl.arange(0, CHUNK_SIZE)[:, None] * row_stride
    col_range = (
        tl.arange(0, BLOCK_D)[None, :] * col_stride
    )  # not needed, since should be contiguous along feature dim

    pid_batch, pid_d, pid_chunk_start = get_program_ids(
        start_pid,
        tiles_per_seq,
        d_tiles_per_chunk,
        effective_chunks_per_seq,  # chunks_per_seq
    )
    pid_chunk_start *= CHUNK_TILES_PER_PROGRAM

    # First determine offset by batch
    batch_offset = pid_batch * batch_stride
    # Next determine offset by chunk
    # offset along feature dim (d)
    col_offset = pid_d * BLOCK_D
    # Map col_offset to filter group
    filter_group = col_offset // dg

    T = load_toeplitz(
        h_ptr,
        FILTER_LEN,
        CHUNK_SIZE,
        SINGLE_GROUP=SINGLE_GROUP,
        group_num=filter_group,
    )

    # Each program processes CHUNK_TILES_PER_PROGRAM chunks
    # for chunk_iter in tl.static_range(CHUNK_TILES_PER_PROGRAM):
    for chunk_iter in tl.static_range(CHUNK_TILES_PER_PROGRAM):
        # for chunk_iter in tl.range(CHUNK_TILES_PER_PROGRAM, num_stages=0):
        pid_chunk = pid_chunk_start + chunk_iter

        if ENABLE_CHECK:
            if pid_chunk > chunks_per_seq - 1:
                break

        chunk_offset = pid_chunk * chunk_stride
        offsets = batch_offset + chunk_offset + col_offset + row_range + col_range

        # tl.static_print("pid", start_pid, "pid_batch", pid_batch, "pid_d", pid_d, "pid_chunk", pid_chunk)

        x = tl.load(x_ptr + offsets)
        B = tl.load(B_ptr + offsets)
        C = tl.load(C_ptr + offsets)

        Bx = B * x

        y = tl.dot(
            T,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=out_dtype,
        )

        if RETURN_TOEPLITZ:
            blocks_per_filter_group = dg // BLOCK_D
            is_first_pid_in_group = (pid_d % blocks_per_filter_group) == 0
            t_group_stride = CHUNK_SIZE * CHUNK_SIZE
            t_offsets = (
                filter_group * t_group_stride
                + tl.arange(0, CHUNK_SIZE)[:, None] * CHUNK_SIZE
                + tl.arange(0, CHUNK_SIZE)[None, :]
            )
            # Only need to store once per batch per seq per filter group
            if pid_batch == 0 and pid_chunk == 0:
                # Only first program for each filter group needs to store
                if is_first_pid_in_group:
                    tl.store(T_ptr + t_offsets, T)

        # Now load B_lag and x_lag
        # First chunk (t = 0) does not need correction
        if pid_chunk > 0:
            T_c = load_correction_toeplitz(
                h_ptr,
                FILTER_LEN,
                CHUNK_SIZE,
                SINGLE_GROUP=SINGLE_GROUP,
                group_num=filter_group,
            )

            lag_offsets = offsets - chunk_stride
            B_lag = tl.load(B_ptr + lag_offsets)
            x_lag = tl.load(x_ptr + lag_offsets)
            Bx_lag = B_lag * x_lag

            if RETURN_BX_LAG:
                tl.store(bx_lag_ptr + lag_offsets, Bx_lag)

            correction_term = tl.dot(
                T_c,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=out_dtype,
            )

            y += correction_term

            if RETURN_TOEPLITZ:
                # First chunk doesn't calculate correction toeplitz
                # pid_batch = 0, pid_chunk = 1 is the first batch / chunk that calculates correction term
                if pid_batch == 0 and pid_chunk == 1:
                    if is_first_pid_in_group:
                        tl.store(T_hat_ptr + t_offsets, T_c)

        if RETURN_Y2:
            tl.store(y2_ptr + offsets, y)

        y *= C
        out_idx = out_ptr + offsets
        tl.store(out_idx, y)


# _default_grouped_autotuner = triton.autotune(
#     configs=get_autotune_configs(),
#     key=["bs", "seqlen", "g", "dg"],
# )
# _persistent_grouped_autotuner = triton.autotune(
#     configs=get_persistent_configs(),
#     key=["bs", "seqlen", "g", "dg"],
# )

# _two_pass_grouped_heuristic = get_two_pass_heuristics()
# _two_pass_fwd_grouped_persistent_autotuned = _persistent_grouped_autotuner(
#     _two_pass_grouped_heuristic(_two_pass_fwd_grouped_kernel)
# )
# _two_pass_fwd_grouped_default_autotuned = _default_grouped_autotuner(
#     _two_pass_grouped_heuristic(_two_pass_fwd_grouped_kernel)
# )


# # TODO: precompile kernel for persistent schedule to maximize occupancy
# def two_pass_fwd_grouped(
#     x: torch.Tensor,
#     B: torch.Tensor,
#     C: torch.Tensor,
#     h: torch.Tensor,
#     y: torch.Tensor = None,
#     return_y2: bool = False,  # y2 = T_local @ B*x + T_hat @ B_lag*x_lag
#     return_toeplitz: bool = False,
#     verbose: bool = False,
#     schedule: str = "default",
#     autotune: bool = False,
#     version: str = "v1",
#     CHUNK_SIZE: int = None,
#     BLOCK_D: int = None,
#     CHUNK_TILES_PER_PROGRAM: int = 1,
#     NUM_PIPELINE_STAGES: int = 0,  # for tl.range
#     THREADBLOCK_SWIZZLE: str = "row",
#     USE_TMA: bool = False,
#     num_warps: int = 4,
#     # TODO: Make sure to set match these defaults to those in CUDAOptions
#     num_stages: int = 3,  # for tl.dot, should default to 3
#     num_ctas: int = 1,
#     maxnreg: int = None,
#     warmup: bool = False,
#     return_kernel: bool = False,
#     return_autotune_result: bool = False,
#     DEBUG: bool = False,
# ) -> Union[torch.tensor, tuple[triton.compiler.CompiledKernel, tuple[Any], tuple[Any]]]:
#     """
#     cgcg 2-pass with grouped filters

#     `g`: number of groups along feature dim:
#     `dg`: number of features per group

#     Assumptions:
#         - g == 1: single filter shared among all features
#         - 1 < g < d: `g` groups where each group has `dg` features.  `dg` must be power of 2 and > `16`
#         to leverage tensorcores.
#         - g == d: each feature has its own filter, not implemented currently since this results in GEMV

#     - x, B, C: bs x l x g x dg where g * dg = hidden_dim.
#     - h: g x 1 x hl where hl is the filter length and must fit within chunk_size

#     Args:
#         x (torch.tensor): (bs, l, g, dg)
#         B (torch.tensor): same shape as x
#         C (torch.tensor): same shape as x
#         h (torch.tensor): (g, 1, hl)
#         y (Optional[torch.tensor]): (bs, l, g, dg), pre-allocated output
#         autotune (bool): If true, use autotuning.
#         schedule (str): One of "default" or "persistent":
#         - "default" launches a 1-d grid with num programs == total tiles
#         - "persistent" launches num_programs = min(NUM_SM, total_tiles), the idea being that
#         reuse of CTAs should allow for better pipelining (hiding memory latency).
#         CHUNK_SIZE, BLOCK_D, num_warps, num_stages, NUM_PIPELINE_STAGES: these are for running a manually configured kernel
#         If any are specified, all must be specified.
#         NOTE: NUM_PIPELINE_STAGES is for pipelining `tl.range` as opposed to `num_stages` which is used for GEMM pipelining.
#         warmup (bool): If true, compile the kernel and return the compiled kernel.
#         return_kernel (bool): If true, run and return the compiled kernel.
#         return_autotune_result (bool): If true, return the autotune result.  Only valid if `autotune=True`.
#     Returns:
#         Return type dependent on `warmup`, `return_kernel`, `return_autotune_result`
#         - default is `y` the output tensor with shape (bs, l, g, dg)
#         - if `warmup=True`, then the compiled kernel (triton.compiler.CompiledKernel) along with kernel args and kernel constexprs are returned
#         - if `return_kernel=True`, then the `y` is returned along with the kernel (triton.runtime.JITFunction)
#         - if `return_autotune_result=True`, then a 2-tuple of `y` and the autotuned result (see AutotunedResult) is returned
#     """
#     bs, seqlen, g, dg = x.shape

#     # basic shape checks
#     assert dg >= 16, "dg must be >= 8 to use tensor-cores"
#     assert x.shape == B.shape == C.shape
#     hg, _in_channel_div_group, filter_len = h.shape
#     assert hg == g
#     assert _in_channel_div_group == 1

#     # hidden_dim
#     d = g * dg

#     x = x.reshape(bs, seqlen, d)
#     B = B.reshape_as(x)
#     C = C.reshape_as(x)
#     batch_stride, row_stride, col_stride = x.stride()

#     # triton kernel pre-condition
#     assert x.is_contiguous()
#     assert B.is_contiguous()
#     assert C.is_contiguous()

#     # Reshape h to a 2-D tensor
#     # TODO: remove?
#     h = h.reshape(g, filter_len)
#     assert h.is_contiguous()

#     # use_autotuner = not any([CHUNK_SIZE, BLOCK_D, num_warps, NUM_PIPELINE_STAGES])
#     assert not (
#         autotune and warmup
#     ), "autotune and warmup are not supported, use return_kernel=True to get the kernel after autotuning"

#     if autotune:
#         assert not USE_TMA, "TMA is not supported for autotuning"
#         if schedule == "default":
#             kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_default_autotuned
#         elif schedule == "persistent":
#             kernel: triton.runtime.JITFunction = (
#                 _two_pass_fwd_grouped_persistent_autotuned
#             )
#     else:
#         assert all(
#             [
#                 CHUNK_SIZE,
#                 BLOCK_D,
#                 # num_warps,
#                 # num_stages is not None,
#             ]
#         ), "Must specify all of CHUNK_SIZE, BLOCK_D, NUM_PIPELINE_STAGES"
#         if USE_TMA:
#             kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_tma_kernel
#         elif version == "v1":
#             assert NUM_PIPELINE_STAGES is not None, "Must specify NUM_PIPELINE_STAGES for version v1"
#             kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_kernel
#         elif version == "v2":
#             assert CHUNK_TILES_PER_PROGRAM is not None
#             assert schedule == "default", "schedule must be default for version v2"
#             kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_kernel_v2
#         else:
#             raise ValueError(f"Unknown version: {version}")

#     if BLOCK_D is not None:
#         assert dg % BLOCK_D == 0, "dg must be multiple of BLOCK_D"

#     if CHUNK_TILES_PER_PROGRAM is not None and CHUNK_SIZE is not None:
#         assert triton.cdiv(seqlen, CHUNK_SIZE) % CHUNK_TILES_PER_PROGRAM == 0

#     if schedule == "default":
#         if version == "v2":
#             def _1d_grid(META):
#                 row_tiles = triton.cdiv(seqlen, META["CHUNK_SIZE"])
#                 # Each program processes CHUNK_TILES_PER_PROGRAM tiles
#                 # assert row_tiles % META["CHUNK_TILES_PER_PROGRAM"] == 0
#                 grid_chunks = triton.cdiv(row_tiles, META["CHUNK_TILES_PER_PROGRAM"])

#                 col_tiles = triton.cdiv(d, META["BLOCK_D"])
#                 # total_tiles = bs * row_tiles * col_tiles
#                 total_programs = bs * grid_chunks * col_tiles

#                 return (total_programs,)
#         else:
#             def _1d_grid(META):
#                 row_tiles = triton.cdiv(seqlen, META["CHUNK_SIZE"])
#                 col_tiles = triton.cdiv(d, META["BLOCK_D"])
#                 total_programs = bs * row_tiles * col_tiles

#                 return (total_programs,)

#         NUM_PIPELINE_STAGES = 0
#         grid = _1d_grid

#     elif schedule == "persistent":
#         grid = lambda META: (
#             min(
#                 NUM_SM,
#                 triton.cdiv(seqlen, META["CHUNK_SIZE"])
#                 * triton.cdiv(d, META["BLOCK_D"])
#                 * bs,
#             ),
#         )
#     else:
#         raise ValueError(f"schedule {schedule} not implemented")

#     if y is None:
#         y = torch.zeros_like(x)

#     # For backwards
#     if return_y2:
#         y2 = torch.empty_like(x)
#     else:
#         y2 = None

#     if return_toeplitz:
#         assert (
#             CHUNK_SIZE is not None
#         ), "CHUNK_SIZE must be specified for return_toeplitz"
#         # NOTE: Need to initialize T_hat as zeros, since not all chunks need correction term
#         T = torch.zeros(g, CHUNK_SIZE, CHUNK_SIZE, device=x.device, dtype=x.dtype)
#         T_hat = torch.zeros_like(T)
#     else:
#         T = None
#         T_hat = None

#     if verbose:
#         print(f"{x.shape=}, {B.shape=}, {C.shape=}, {h.shape=}, {y.shape=}")
#         print(f"{bs=} {seqlen=} {g=} {dg=} {filter_len=}")
#         print(f"{CHUNK_SIZE=}, {BLOCK_D=}, {num_warps=}, {NUM_PIPELINE_STAGES=}")

#     kernel_args = (
#         x,
#         B,
#         C,
#         h,
#         y,
#         T,
#         T_hat,
#         y2,
#         batch_stride,
#         row_stride,
#         col_stride,
#         bs,
#         seqlen,
#         g,
#         dg,
#     )

#     kernel_constexprs = {
#         "FILTER_LEN": filter_len,
#         "SINGLE_GROUP": g == 1,
#         "RETURN_TOEPLITZ": return_toeplitz,
#         "RETURN_Y2": return_y2,

#     }
#     if not autotune:
#         kernel_constexprs.update(
#             {
#                 "CHUNK_SIZE": CHUNK_SIZE,
#                 "BLOCK_D": BLOCK_D,
#                 "THREADBLOCK_SWIZZLE": THREADBLOCK_SWIZZLE,
#                 "num_warps": num_warps,
#                 "num_stages": num_stages,
#                 "num_ctas": num_ctas,
#             })
#         if version == "v1":
#             kernel_constexprs.update(
#                 {
#                     "NUM_PIPELINE_STAGES": NUM_PIPELINE_STAGES,
#                 }
#             )
#             if USE_TMA:
#                 kernel_constexprs.update({"DTYPE": x.dtype})
#         else:
#             kernel_constexprs.update(
#                 {
#                     "CHUNK_TILES_PER_PROGRAM": CHUNK_TILES_PER_PROGRAM,
#                 }
#             )

#     if warmup:
#         compiled_kernel: triton.compiler.CompiledKernel = kernel.warmup(
#             *kernel_args, **kernel_constexprs, grid=(1,)
#         )
#         return compiled_kernel, kernel_args, kernel_constexprs
#     else:
#         compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](
#             *kernel_args, **kernel_constexprs
#         )

#         y = y.reshape(bs, seqlen, g, dg)
#         if y2 is not None:
#             y2 = y2.reshape(bs, seqlen, g, dg)

#         # If no auxiliary results requested, just return tensors
#         if not return_autotune_result:
#             if return_kernel:
#                 return y, T, T_hat, y2, compiled_kernel
#             return y, T, T_hat, y2

#         if return_autotune_result:
#             keys = [
#                 k for k in kernel.cache.keys() if kernel.cache[k] == kernel.best_config
#             ]
#             # Filter for best key, as best_config can be the same for multiple keys
#             # TODO: improve this since this is a bit hacky
#             # Key is best key if the kernel args match those of the current kernel args and the dtype is the same
#             # Assumption is that dtype is the same for all inputs and output
#             best_key = [
#                 k
#                 for k in keys
#                 if k[: len(kernel.key_idx)] == (bs, seqlen, g, dg)
#                 and k[len(kernel.key_idx)] == str(x.dtype)
#             ]
#             assert len(best_key) == 1
#             # print(f"Autotune Best Config {kernel.best_config} for keys {best_key}")
#             autotune_result = AutotunedResult(
#                 best_config=kernel.best_config, key=best_key[0]
#             )

#             if return_kernel:
#                 return y, T, T_hat, y2, compiled_kernel, autotune_result

#             return y, T, T_hat, y2, autotune_result


# if __name__ == "__main__":
#     from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected
#     from savanna.kernels.triton_src.cgcg.triton.fwd_tma import two_pass_fwd_grouped_tma
#     from savanna.kernels.triton_src.cgcg.triton.kernel_utils import get_kernel_occupancy
#     from savanna.kernels.triton_src.cgcg.utils import correction_toeplitz, toeplitz
#     # Debugging settings -> max precision, set seed
#     dtype = torch.float32  # torch.float16 leads to numerical differences
#     torch.set_float32_matmul_precision("highest")
#     device = "cuda"
#     torch.manual_seed(0)

#     # Shapes
#     bs = 1
#     seqlen = 1024
#     hl = 128  # Filter size
#     d = 768
#     g = 2
#     dg = d // g
#     # Only for debugging
#     CHUNK_SIZE = max(hl, 32)  # seqlen // 4
#     BLOCK_D = 32  # if dg > 32 else dg
#     CHUNK_TILES_PER_PROGRAM = 4
#     x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)

#     # y_ref = gcg_fwd_ref_debug(x, B, C, h, interleave=True)
#     h_ = h.flip(-1)
#     # T = toeplitz(h_[:, 0], chunk_size)
#     # T_c = correction_toeplitz(h_[:, 0], chunk_size)

#     # print(f"{target=} {NUM_SM=} {NUM_REGS=} {SIZE_SMEM=} {WARP_SIZE=}")
#     num_warps = 4
#     swizzle = "row"
#     warmup = False
#     autotune = False
#     schedule = "default"
#     return_kernel = True
#     return_autotune_result = False
#     return_toeplitz = False
#     return_y2 = False

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
#             "DEBUG": False,
#         }

#     y_v1, *_ = two_pass_fwd_grouped(
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
#         **kernel_config_v1,
#     )

#     y_v1_tma, *_ = two_pass_fwd_grouped_tma(
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
#         # return_autotune_result=return_autotune_result,
#         return_toeplitz=return_toeplitz,
#         return_y2=return_y2,
#         **kernel_config_v1,
#     )
#     y_v2, *_, kernel = two_pass_fwd_grouped(
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
#         **kernel_config_v2,
#     )
#     # ptx = kernel.asm["ptx"]
#     # with open("fwd_v2.ptx", "w") as f:
#     #     f.write(ptx)
#     y_ref = gcg_fwd_ref_corrected(x, B, C, h, use_causal_conv=False)
#     v1_diff = (y_ref - y_v1).abs().max()
#     v1_tma_diff = (y_ref - y_v1_tma).abs().max()
#     v2_diff = (y_ref - y_v2).abs().max()
#     print(f"{v1_diff=}")
#     print(f"{v1_tma_diff=}")
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
