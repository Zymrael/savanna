import itertools
import math

import triton
import triton.language as tl

from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    get_program_ids,
    is_power_of_2,
)
from savanna.kernels.triton_src.cgcg.src.toeplitz_kernels import (
    load_correction_toeplitz,
    load_toeplitz,
)


def get_autotune_configs(
    warps=[2, 4, 8],
    stages=[1, 2, 3, 4, 5],
    chunk_tile_sizes=[32, 64, 128, 256],
    d_block_tile_sizes=[32, 64, 128, 256],
    threadblock_swizzle=["row", "col"],
    chunk_tiles_per_program=[1, 2, 3, 4],
):
    configs = []

    configs = [
        triton.Config(
            {
                "CHUNK_SIZE": c,
                "BLOCK_D": d,
                "THREADBLOCK_SWIZZLE": swz,
                "CHUNK_TILES_PER_PROGRAM": chunk_tiles,
            },
            num_warps=w,
            num_stages=stg,
        )
        for w, stg, c, d, swz, chunk_tiles in itertools.product(
            warps, stages, chunk_tile_sizes, d_block_tile_sizes, threadblock_swizzle, chunk_tiles_per_program
        )
    ]

    return configs


# def get_autotune_configs(
#     min_warps=4,
#     max_warps=8,
#     min_stages=1,
#     max_stages=5,
#     min_chunk_size=32,
#     max_chunk_size=128,
#     min_d_block=32,
#     max_d_block=128,
#     threadblock_swizzle=["row", "col"],
#     chunk_tiles_per_program=[1, 2],
# ):
#     configs = []
#     assert is_power_of_2(min_warps) and is_power_of_2(max_warps)
#     assert min_chunk_size % 16 == 0 and max_chunk_size % 16 == 0
#     warp_range = [2**i for i in range(int(math.log2(min_warps)), int(math.log2(max_warps) + 1))]
#     stage_range = list(range(min_stages, max_stages + 1))
#     chunk_range = [2**i for i in range(int(math.log2(min_chunk_size)), int(math.log2(max_chunk_size) + 1))]
#     d_block_range = [2**i for i in range(int(math.log2(min_d_block)), int(math.log2(max_d_block) + 1))]

#     configs = [
#         triton.Config(
#             {
#                 "CHUNK_SIZE": c,
#                 "BLOCK_D": d,
#                 "THREADBLOCK_SWIZZLE": swz,
#                 "CHUNK_TILES_PER_PROGRAM": chunk_tiles,
#             },
#             num_warps=w,
#             num_stages=stg,
#         )
#         for w, stg, c, d, swz, chunk_tiles in itertools.product(
#             warp_range, stage_range, chunk_range, d_block_range, threadblock_swizzle, chunk_tiles_per_program
#         )
#     ]

#     return configs


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


@triton.jit
def _two_pass_fwd_refactor_kernel(
    # Input tensors
    x_ptr,
    B_ptr,
    C_ptr,
    h_ptr,
    # Intermediate activations
    bx_ptr,
    y2_ptr,
    y_ptr,
    # Strides
    batch_stride,
    row_stride,
    col_stride,
    # Shapes
    bs,
    seqlen,
    g,
    dg,
    # Compile-time constants
    FILTER_LEN: tl.constexpr,
    # Autotuned params
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    # Set by heuristic
    SINGLE_GROUP: tl.constexpr,
    CHUNK_TILES_PER_PROGRAM: tl.constexpr,  # = 1,
    ENABLE_CHECK: tl.constexpr = False,
    # Intermediates for Bwd
    RETURN_BX: tl.constexpr = True,
    RETURN_Y2: tl.constexpr = True,
    DEBUG: tl.constexpr = False,
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

        # Store for backwards pass
        if RETURN_BX:
            tl.store(bx_ptr + offsets, Bx)

        y = tl.dot(
            T,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=out_dtype,
        )
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

            correction_term = tl.dot(
                T_c,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=out_dtype,
            )

            y += correction_term

        # Store for backwards pass
        if RETURN_Y2:
            tl.store(y2_ptr + offsets, y)

        y *= C
        out_idx = y_ptr + offsets

        # Final output
        tl.store(out_idx, y)


_default_grouped_autotuner = triton.autotune(
    configs=get_autotune_configs(),
    key=["bs", "seqlen", "g", "dg"],
)
# _persistent_grouped_autotuner = triton.autotune(
#     configs=get_persistent_configs(),
#     key=["bs", "seqlen", "g", "dg"],
# )

_two_pass_grouped_heuristic = get_two_pass_heuristics()
# _two_pass_fwd_grouped_persistent_autotuned = _persistent_grouped_autotuner(
#     _two_pass_grouped_heuristic(_two_pass_fwd_grouped_kernel)
# )
_two_pass_fwd_refactor_autotuned = _default_grouped_autotuner(
    _two_pass_grouped_heuristic(_two_pass_fwd_refactor_kernel)
)