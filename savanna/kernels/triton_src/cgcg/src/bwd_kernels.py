from enum import Enum

import triton
import triton.language as tl
from arrow import get

from savanna.kernels.triton_src.cgcg.src.kernel_utils import (  # DeviceProps,
    get_program_ids,
)
from savanna.kernels.triton_src.cgcg.src.toeplitz_kernels import (
    load_correction_toeplitz,
    load_toeplitz,
)

# logger = logging.getLogger(__name__)


class Direction(Enum):
    LOWER = 0
    UPPER = 1


SUB_DIAG = tl.constexpr(Direction.LOWER.value)
SUPER_DIAG = tl.constexpr(Direction.UPPER.value)


@triton.jit
def generate_diag_mask(offset, CHUNK_SIZE: tl.constexpr, DIR: tl.constexpr):
    assert (DIR == SUB_DIAG) or (DIR == SUPER_DIAG)

    if DIR == SUB_DIAG:
        offset = -offset

    row_idx = tl.arange(0, CHUNK_SIZE)[:, None] + offset
    col_idx = tl.arange(0, CHUNK_SIZE)[None, :]
    mask = row_idx == col_idx
    return mask


@triton.jit
def diag_sum(
    T,
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DIR: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    out = tl.zeros((FILTER_LEN,), dtype=T.dtype)
    idx_range = tl.arange(0, FILTER_LEN)

    if DIR == SUPER_DIAG:
        start: tl.constexpr = CHUNK_SIZE - (FILTER_LEN - 1)
        end: tl.constexpr = CHUNK_SIZE
    else:
        start: tl.constexpr = 0
        end: tl.constexpr = FILTER_LEN
    if DEBUG:
        tl.static_print("DIR", DIR)
        tl.static_print("start", start)

    for os in tl.static_range(start, end):
        mask = generate_diag_mask(os, CHUNK_SIZE=CHUNK_SIZE, DIR=DIR)
        masked_T = tl.where(mask, T, 0)
        summed_T = tl.sum(masked_T)
        if DIR == SUB_DIAG:
            out_idx = FILTER_LEN - 1 - os
        else:
            out_idx = os - start

        idx_mask = idx_range == out_idx
        out = tl.where(idx_mask, summed_T, out)
        if DEBUG:
            tl.static_print("mask\n", mask)
            tl.static_print("masked x\n", masked_T)
            tl.static_print("idx mask\n", out_idx)
            # tl.static_print("os", os, "out\n", out)

    return out


@triton.jit
def _diag_to_row(CHUNK_SIZE: tl.constexpr, DIRECTION: tl.constexpr = "lower"):
    rows = tl.arange(0, CHUNK_SIZE)[None, :]
    cols = tl.arange(0, CHUNK_SIZE)[:, None]
    if DIRECTION == "lower":
        row_idx = cols - rows
    else:
        row_idx = rows - cols
    return row_idx


@triton.jit
def _get_T_store_idx(
    CHUNK_SIZE: tl.constexpr,
    FILTER_LEN: tl.constexpr,
    row_stride,
    col_stride,
    DEBUG: tl.constexpr = False,
):
    row_idx = _diag_to_row(CHUNK_SIZE)
    row_idx = FILTER_LEN - 1 - row_idx
    col_idx = tl.arange(0, CHUNK_SIZE)[None, :]
    offsets = row_idx * row_stride + col_idx * col_stride
    # Mask sub-diagonal slice of dT containing main diagonal and sub-diagonals up to FILTER_LEN
    # Lower triangular mask
    tril_mask = tl.arange(0, CHUNK_SIZE)[:, None] >= col_idx
    lower_mask = (tl.arange(0, CHUNK_SIZE)[:, None] - FILTER_LEN) < col_idx
    # filter_mask = tl.arange(0, CHUNK_SIZE)[:, None] < FILTER_LEN
    # tl.static_print("filter_mask", filter_mask)
    mask = tril_mask & lower_mask

    store_idx = tl.where(mask, offsets, 0)
    if DEBUG:
        tl.static_print("row_idx\n", row_idx)
        tl.static_print("offsets\n", offsets)
        tl.static_print("tril_mask\n", tril_mask)
        tl.static_print("lower_mask\n", lower_mask)
        tl.static_print("mask\n", mask)
        tl.static_print("store_idx\n", store_idx)

    return store_idx, mask


@triton.jit
def store_T_kernel(
    dT_ptr,
    h_ptr,
    group_stride,
    row_stride,
    col_stride,
    CHUNK_SIZE: tl.constexpr,
    FILTER_LEN: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    """
    Generates offsets for storing the gradient toeplitz matrix such that diagonals and sub-diagonals are stored
    as contiguous rows, ordered by filter index.

    E.g., for CHUNK_SIZE = 4 and FILTER_LEN = 4,
        [4, 0, 0, 0]
        [3, 4, 0, 0]
        [2, 3, 4, 0]
        [1, 2, 3, 4]
    will be stored as
        [1, 0, 0, 0]
        [2, 2, 0, 0]
        [3, 3, 3, 0]
        [4, 4, 4, 4]
    Args:
        dT_ptr: Pointer to the gradient tensor of shape (CHUNK_SIZE, CHUNK_SIZE).
        h_ptr: Pointer to the output gradient filter buffer of shape (FILTER_LEN, CHUNK_SIZE).

    The reason for doing this is so that the gradients for the filter can then be efficiently computed
    in the subsequent kernel.
    """
    pid = tl.program_id(0)
    # Each program handles a filter group
    store_idx, mask = _get_T_store_idx(CHUNK_SIZE, FILTER_LEN, row_stride, col_stride, DEBUG=DEBUG)

    # dT group stride is CHUNK_SIZE * CHUNK_SIZE
    load_offset = pid * CHUNK_SIZE * CHUNK_SIZE
    load_idx = tl.arange(0, CHUNK_SIZE)[:, None] * row_stride + tl.arange(0, CHUNK_SIZE)[None, :] * col_stride
    T = tl.load(dT_ptr + load_offset + load_idx)

    store_offset = pid * FILTER_LEN * CHUNK_SIZE
    tl.store(h_ptr + store_offset + store_idx, T, mask=mask)


@triton.jit
def _get_Tc_store_idx(
    CHUNK_SIZE: tl.constexpr,
    FILTER_LEN: tl.constexpr,
    row_stride,
    col_stride,
    DEBUG: tl.constexpr = False,
):
    OFFSET: tl.constexpr = min(FILTER_LEN - 1 - CHUNK_SIZE, 0)
    row_idx = _diag_to_row(CHUNK_SIZE, DIRECTION="upper")
    row_idx = row_idx + OFFSET

    col_idx = tl.arange(0, CHUNK_SIZE)[None, :]

    # Keep only super-diagonals starting from CHUNK_SIZE - FILTER_LEN - 1
    triu_mask = tl.arange(0, CHUNK_SIZE)[:, None] - OFFSET <= col_idx

    offsets = row_idx * row_stride + col_idx * col_stride
    store_idx = tl.where(triu_mask, offsets, 0)

    if DEBUG:
        tl.static_print("row_idx\n", row_idx)
        tl.static_print("OFFSET\n", OFFSET)
        tl.static_print("triu_mask\n", triu_mask)
        tl.static_print("store_idx\n", store_idx)

    return store_idx, triu_mask


@triton.jit
def store_Tc_kernel(
    dTc_ptr,
    h_ptr,
    group_stride,
    row_stride,
    col_stride,
    CHUNK_SIZE: tl.constexpr,
    FILTER_LEN: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    """
    Generates offsets for storing the gradient toeplitz correction matrix such that diagonals and super-diagonals are stored
    as contiguous rows, ordered by filter index.

    E.g., for CHUNK_SIZE = 4 and FILTER_LEN = 4,
        [0, 1, 2, 3]
        [0, 0, 1, 2]
        [0, 0, 0, 1]
        [0, 0, 0, 0]
    will be stored as
        [0, 1, 1, 1]
        [0, 0, 2, 2]
        [0, 0, 0, 3]
        [0, 0, 0, 0]

    The reason for doing this is so that the gradients for the filter can then be efficiently computed
    in the subsequent kernel.
    """
    # Each program handles a filter group
    pid = tl.program_id(0)

    store_idx, mask = _get_Tc_store_idx(CHUNK_SIZE, FILTER_LEN, row_stride, col_stride, DEBUG=DEBUG)

    # Group stride is CHUNK_SIZE * CHUNK_SIZE
    load_offset = pid * CHUNK_SIZE * CHUNK_SIZE
    load_idx = tl.arange(0, CHUNK_SIZE)[:, None] * row_stride + tl.arange(0, CHUNK_SIZE)[None, :] * col_stride
    Tc = tl.load(dTc_ptr + load_offset + load_idx)

    store_offset = pid * FILTER_LEN * CHUNK_SIZE
    tl.store(h_ptr + store_offset + store_idx, Tc, mask=mask)


@triton.jit
def _two_pass_bwd_grouped_kernel_v1(
    # Inputs, saved from fwd
    dy_ptr,
    x_ptr,
    B_ptr,
    C_ptr,
    h_ptr,
    # Intermediate activations
    y2_ptr,  # TODO: rename
    # Optionally loaded Toeplitz matrices
    T_ptr,
    T_hat_ptr,
    bx_lag_ptr,
    # Output ptrs
    dx_ptr,
    dB_ptr,
    dC_ptr,
    dhdT_ptr,
    dhdTc_ptr,
    # Strides
    input_batch_stride,
    input_row_stride,
    input_col_stride,
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
    # Compile-time constants
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    # Whether to load toeplitz matrices
    LOAD_T: tl.constexpr,
    LOAD_BX_LAG: tl.constexpr,
    SINGLE_GROUP: tl.constexpr,
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
    # No need to set here since these can be passed directly to kernel call
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    # num_ctas: tl.constexpr = 1,
    DEBUG: tl.constexpr = False,
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
                "LOAD_T:",
                LOAD_T,
            )

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

    row_range = tl.arange(0, CHUNK_SIZE)[:, None] * input_row_stride
    col_range = (
        tl.arange(0, BLOCK_D)[None, :] * input_col_stride
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

        load_offsets = batch_offset + chunk_offset + col_offset + row_range + col_range
        x = tl.load(x_ptr + load_offsets)
        # Reuse offsets for B, C, dy, y2
        B = tl.load(B_ptr + load_offsets)
        C = tl.load(C_ptr + load_offsets)
        dy = tl.load(dy_ptr + load_offsets)
        y2 = tl.load(y2_ptr + load_offsets)

        # Start backprop
        dC = dy * y2
        # Backprop through C
        dy = dy * C

        if LOAD_T:
            T_group_stride = CHUNK_SIZE * CHUNK_SIZE
            T_group_offset = filter_group * T_group_stride
            T_idx = tl.arange(0, CHUNK_SIZE)[:, None] * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)[None, :]
            T = tl.load(T_ptr + T_group_offset + T_idx)
        else:
            T = load_toeplitz(
                h_ptr,
                FILTER_LEN,
                CHUNK_SIZE,
                SINGLE_GROUP=SINGLE_GROUP,
                group_num=filter_group,
            )

        T = tl.trans(T)
        dy1 = tl.dot(
            T,
            dy,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=dy.dtype,
        )
        dx = dy1 * B
        dB = dy1 * x

        # Gradient wrt h_local
        Bx = tl.trans(B * x)
        dT = tl.dot(
            dy,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=dy.dtype,
        )

        # Correction term
        # In backwards, we roll in the opposite direction
        # Hence, the last chunk in the sequence does not need correction
        is_last_chunk = pid_chunk == chunks_per_seq - 1
        is_first_chunk = pid_chunk == 0

        if not is_last_chunk:
            dy_lead_idx = dy_ptr + load_offsets + chunk_stride
            dy_lead = tl.load(dy_lead_idx)
            C_lead_idx = C_ptr + load_offsets + chunk_stride
            C_lead = tl.load(C_lead_idx)
            dy_lead *= C_lead

            if LOAD_T:
                # offset and idx defined above
                T_c = tl.load(T_hat_ptr + T_group_offset + T_idx)
            else:
                T_c = load_correction_toeplitz(
                    h_ptr,
                    FILTER_LEN,
                    CHUNK_SIZE,
                    SINGLE_GROUP=SINGLE_GROUP,
                    group_num=filter_group,
                )
            T_c = tl.trans(T_c)

            dcorrection = tl.dot(
                T_c,
                dy_lead,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=dy.dtype,
            )
            dcorrection_dx = dcorrection * B
            dcorrection_dB = dcorrection * x
            dx += dcorrection_dx
            dB += dcorrection_dB

        store_offsets = batch_offset + chunk_offset + col_offset + row_range + col_range

        tl.store(dx_ptr + store_offsets, dx)
        tl.store(dB_ptr + store_offsets, dB)
        tl.store(dC_ptr + store_offsets, dC)

        dhdT_idx, dhdT_mask = _get_T_store_idx(
            CHUNK_SIZE, FILTER_LEN, row_stride=dhdT_row_stride, col_stride=1
        )

        dhdT_offsets = (
            pid_batch * dhdT_batch_stride
            + pid_chunk * dhdT_chunk_stride
            + pid_d * dhdT_block_stride
            + dhdT_idx
        )
        tl.store(dhdT_ptr + dhdT_offsets, dT, mask=dhdT_mask)

        if not is_first_chunk:
            lag_offsets = load_offsets - chunk_stride

            if LOAD_BX_LAG:
                Bx_lag = tl.load(bx_lag_ptr + lag_offsets)
            else:
                B_lag_idx = B_ptr + lag_offsets
                B_lag = tl.load(B_lag_idx)
                x_lag_idx = x_ptr + lag_offsets
                x_lag = tl.load(x_lag_idx)
                Bx_lag = B_lag * x_lag

            Bx_lag = tl.trans(Bx_lag)
            dTc = tl.dot(
                dy,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=dy.dtype,
            )

            dhdTc_idx, dhdTc_mask = _get_Tc_store_idx(
                CHUNK_SIZE, FILTER_LEN, row_stride=dhdT_row_stride, col_stride=1
            )
            dhdTc_offsets = (
                pid_batch * dhdT_batch_stride
                + pid_chunk * dhdT_chunk_stride
                + pid_d * dhdT_block_stride
                + dhdTc_idx
            )
            tl.store(dhdTc_ptr + dhdTc_offsets, dTc, mask=dhdTc_mask)


@triton.jit
def _two_pass_bwd_grouped_kernel_v2(
    # Inputs, saved from fwd
    dy_ptr,
    x_ptr,
    B_ptr,
    C_ptr,
    h_ptr,
    # Intermediate activations
    y2_ptr,  # TODO: rename
    # Optionally loaded Toeplitz matrices
    T_ptr,
    T_hat_ptr,
    bx_lag_ptr,
    # Output ptrs
    dx_ptr,
    dB_ptr,
    dC_ptr,
    dhdT_ptr,
    dhdTc_ptr,
    # Strides
    input_batch_stride,
    input_row_stride,
    input_col_stride,
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
    # Compile-time constants
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    # Whether to load toeplitz matrices
    LOAD_T: tl.constexpr,
    SINGLE_GROUP: tl.constexpr,
    CHUNK_TILES_PER_PROGRAM: tl.constexpr = 1,
    ENABLE_CHECK: tl.constexpr = False,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
    LOAD_BX_LAG: tl.constexpr = False,
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
    # No need to set here since these can be passed directly to kernel call
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    # num_ctas: tl.constexpr = 1,
    DEBUG: tl.constexpr = False,
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
                "LOAD_T:",
                LOAD_T,
            )

    # Map 1D grid to 3D logical coordinates
    hidden_dim = g * dg
    chunks_per_seq = tl.cdiv(seqlen, CHUNK_SIZE)
    effective_chunks_per_seq = tl.cdiv(chunks_per_seq, CHUNK_TILES_PER_PROGRAM)
    d_tiles_per_chunk = tl.cdiv(hidden_dim, BLOCK_D)
    tiles_per_seq = chunks_per_seq * d_tiles_per_chunk
    chunk_stride = CHUNK_SIZE * hidden_dim
    batch_stride = chunk_stride * chunks_per_seq
    total_tiles = bs * tiles_per_seq

    # Grid stride
    start_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    row_range = tl.arange(0, CHUNK_SIZE)[:, None] * input_row_stride
    col_range = (
        tl.arange(0, BLOCK_D)[None, :] * input_col_stride
    )  # not needed, since should be contiguous along feature dim

    pid_batch, pid_d, pid_chunk_start = get_program_ids(
        start_pid,
        tiles_per_seq,
        d_tiles_per_chunk,
        effective_chunks_per_seq,  # chunks_per_seq
    )
    pid_chunk_start *= CHUNK_TILES_PER_PROGRAM

    batch_offset = pid_batch * batch_stride
    # Next determine offset by chunk
    # offset along feature dim (d)
    col_offset = pid_d * BLOCK_D
    # Map col_offset to filter group
    filter_group = col_offset // dg

    if LOAD_T:
        T_group_stride = CHUNK_SIZE * CHUNK_SIZE
        T_group_offset = filter_group * T_group_stride
        T_idx = tl.arange(0, CHUNK_SIZE)[:, None] * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)[None, :]
        T = tl.load(T_ptr + T_group_offset + T_idx)
    else:
        T = load_toeplitz(
            h_ptr,
            FILTER_LEN,
            CHUNK_SIZE,
            SINGLE_GROUP=SINGLE_GROUP,
            group_num=filter_group,
        )

    T = tl.trans(T)

    for chunk_iter in tl.static_range(CHUNK_TILES_PER_PROGRAM):
        # for chunk_iter in tl.range(CHUNK_TILES_PER_PROGRAM, num_stages=0):
        pid_chunk = pid_chunk_start + chunk_iter

        if ENABLE_CHECK:
            if pid_chunk > chunks_per_seq - 1:
                break

        chunk_offset = pid_chunk * chunk_stride
        load_offsets = batch_offset + chunk_offset + col_offset + row_range + col_range

        x = tl.load(x_ptr + load_offsets)
        # Reuse offsets for B, C, dy, y2
        B = tl.load(B_ptr + load_offsets)
        C = tl.load(C_ptr + load_offsets)
        dy = tl.load(dy_ptr + load_offsets)
        y2 = tl.load(y2_ptr + load_offsets)

        # Start backprop
        dC = dy * y2
        # Backprop through C
        dy = dy * C

        dy1 = tl.dot(
            T,
            dy,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=dy.dtype,
        )
        dx = dy1 * B
        dB = dy1 * x

        # Gradient wrt h_local
        Bx = tl.trans(B * x)
        dT = tl.dot(
            dy,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=dy.dtype,
        )

        # Correction term
        # In backwards, we roll in the opposite direction
        # Hence, the last chunk in the sequence does not need correction
        is_last_chunk = pid_chunk == chunks_per_seq - 1
        is_first_chunk = pid_chunk == 0

        if not is_last_chunk:
            dy_lead_idx = dy_ptr + load_offsets + chunk_stride
            dy_lead = tl.load(dy_lead_idx)
            C_lead_idx = C_ptr + load_offsets + chunk_stride
            C_lead = tl.load(C_lead_idx)
            dy_lead *= C_lead

            if LOAD_T:
                # offset and idx defined above
                T_c = tl.load(T_hat_ptr + T_group_offset + T_idx)
            else:
                T_c = load_correction_toeplitz(
                    h_ptr,
                    FILTER_LEN,
                    CHUNK_SIZE,
                    SINGLE_GROUP=SINGLE_GROUP,
                    group_num=filter_group,
                )
            T_c = tl.trans(T_c)

            dcorrection = tl.dot(
                T_c,
                dy_lead,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=dy.dtype,
            )
            dcorrection_dx = dcorrection * B
            dcorrection_dB = dcorrection * x
            dx += dcorrection_dx
            dB += dcorrection_dB

        store_offsets = batch_offset + chunk_offset + col_offset + row_range + col_range

        tl.store(dx_ptr + store_offsets, dx)
        tl.store(dB_ptr + store_offsets, dB)
        tl.store(dC_ptr + store_offsets, dC)

        dhdT_idx, dhdT_mask = _get_T_store_idx(
            CHUNK_SIZE, FILTER_LEN, row_stride=dhdT_row_stride, col_stride=1
        )

        dhdT_offsets = (
            pid_batch * dhdT_batch_stride
            + pid_chunk * dhdT_chunk_stride
            + pid_d * dhdT_block_stride
            + dhdT_idx
        )
        tl.store(dhdT_ptr + dhdT_offsets, dT, mask=dhdT_mask)

        if not is_first_chunk:

            lag_offsets = load_offsets - chunk_stride
            if LOAD_BX_LAG:
                Bx_lag = tl.load(bx_lag_ptr + lag_offsets)
            else:
                B_lag_idx = B_ptr + lag_offsets
                B_lag = tl.load(B_lag_idx)
                x_lag_idx = x_ptr + lag_offsets
                x_lag = tl.load(x_lag_idx)
                Bx_lag = B_lag * x_lag

            Bx_lag = tl.trans(Bx_lag)
            dTc = tl.dot(
                dy,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=dy.dtype,
            )

            dhdTc_idx, dhdTc_mask = _get_Tc_store_idx(
                CHUNK_SIZE, FILTER_LEN, row_stride=dhdT_row_stride, col_stride=1
            )
            dhdTc_offsets = (
                pid_batch * dhdT_batch_stride
                + pid_chunk * dhdT_chunk_stride
                + pid_d * dhdT_block_stride
                + dhdTc_idx
            )
            tl.store(dhdTc_ptr + dhdTc_offsets, dTc, mask=dhdTc_mask)


def get_bwd_conv_configs():
    block_sizes = [32, 64, 128, 256]
    warps = [2, 4, 8]
    configs = []

    for block_x in block_sizes:
        for block_y in block_sizes:
            for warp in warps:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_X": block_x,
                            "BLOCK_Y": block_y,
                        },
                        num_warps=warp,
                    )
                )
    return configs


@triton.jit
def post_conv_kernel(
    dBx_ptr,
    x_ptr,
    B_ptr,  # input tensors
    dx_ptr,
    dB_ptr,  # output tensors
    bs,
    seqlen,
    d,  # shapes
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """
    dBx_ptr is bs x d x seqlen (output of dconv1d_input)
    x_ptr and B_ptr are bs x seqlen x d

    Load dBx_ptr transposed, elementwise multiply by {x, B} to get dB and dx respectively.
    """
    block_offset_x = tl.program_id(0) * BLOCK_X
    block_offset_y = tl.program_id(1) * BLOCK_Y
    x_index = block_offset_x + tl.arange(0, BLOCK_X)[:, None]
    y_index = block_offset_y + tl.arange(0, BLOCK_Y)[None, :]
    batch_index = block_offset_x // seqlen

    input_row_stride = d
    output_row_stride = seqlen
    batch_stride = d * seqlen

    # Rows are d, cols are seqlen for dBx, flipped for x and B
    load_indices = (x_index % seqlen) * input_row_stride + y_index + (batch_stride * batch_index)
    load_dBx_indices = x_index % seqlen + (y_index % d) * output_row_stride + (batch_stride * batch_index)

    x = tl.load(
        x_ptr + load_indices,
    )
    B = tl.load(
        B_ptr + load_indices,
    )
    dBx = tl.load(dBx_ptr + load_dBx_indices)

    dB = dBx * x
    dx = dBx * B

    tl.store(dx_ptr + load_indices, dx)
    tl.store(dB_ptr + load_indices, dB)


post_conv_kernel_autotune = triton.autotune(
    configs=get_bwd_conv_configs(),
    key=["bs", "seqlen", "d"],
)(post_conv_kernel)


@triton.jit
def pre_conv_kernel(  # input tensors
    dy_ptr,
    C_ptr,
    y2_ptr,
    Bx_ptr,
    # output tensors
    dC_ptr,
    dy2_ptr,
    Bx_permuted_ptr,
    # shapes
    bs,
    seqlen,
    d,
    # constexprs
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """
    - Load dy, y2, and C.
    - Calculate dy2 and dC
    - Store dy2 permuted and dC

    We load tensors from the perspective of dy2 permuted
    - When storing dC, we reuse the load indices --> no permutation
    - When storing dy2, we use permuted indices -> permutation
    """
    # y and x are from perspective of permuted tensor, (bs * n) x m
    # y -> bs * n, x -> m
    yoffset = tl.program_id(1) * BLOCK_Y
    yindex = yoffset + tl.arange(0, BLOCK_Y)[None, :]
    xoffset = tl.program_id(0) * BLOCK_X
    xindex = xoffset + tl.arange(0, BLOCK_X)[:, None]

    input_row_stride = d
    output_row_stride = seqlen
    batch_stride = d * seqlen
    y0 = yindex % input_row_stride
    batch_offset = yindex // input_row_stride

    load_indices = y0 + (input_row_stride * xindex) + (batch_stride * batch_offset)
    store_indices = xindex + (output_row_stride * yindex)
    # Load dy, C, and y2
    dy = tl.load(dy_ptr + load_indices)
    C = tl.load(C_ptr + load_indices)
    y2 = tl.load(y2_ptr + load_indices)
    Bx = tl.load(Bx_ptr + load_indices)
    # Post-gate derivative
    dC = dy * y2

    # Backprop through post-gate to get derivative wrt convolution
    dy2 = dy * C

    # Store dC
    tl.store(dC_ptr + load_indices, dC)
    # Store dy2 and Bx permuted
    tl.store(dy2_ptr + store_indices, dy2)
    tl.store(Bx_permuted_ptr + store_indices, Bx)


pre_conv_kernel_autotune = triton.autotune(
    configs=get_bwd_conv_configs(),
    key=["bs", "seqlen", "d"],
)(pre_conv_kernel)