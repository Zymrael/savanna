import logging
from enum import Enum
from typing import Any, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from savanna.kernels.triton_src.cgcg.src.bwd_kernels import (
    _two_pass_bwd_grouped_kernel_v1,
    _two_pass_bwd_grouped_kernel_v2,
)
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    get_program_ids,
)
from savanna.kernels.triton_src.cgcg.src.toeplitz_kernels import load_correction_toeplitz, load_toeplitz

logger = logging.getLogger(__name__)


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
def _two_pass_bwd_grouped_kernel_wgrad(
    # Inputs, saved from fwd
    dy_ptr,
    x_ptr,
    B_ptr,
    C_ptr,
    # Outputs
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
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
    # No need to set here since these can be passed directly to kernel call
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    # num_ctas: tl.constexpr = 1,
    DEBUG: tl.constexpr = True,
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
        THREADBLOCK_SWIZZLE,
    )
    pid_chunk_start *= CHUNK_TILES_PER_PROGRAM

    batch_offset = pid_batch * batch_stride
    # Next determine offset by chunk
    # offset along feature dim (d)
    col_offset = pid_d * BLOCK_D
    # Map col_offset to filter group
    # filter_group = col_offset // dg

    # if LOAD_T:
    #     T_group_stride = CHUNK_SIZE * CHUNK_SIZE
    #     T_group_offset = filter_group * T_group_stride
    #     T_idx = (
    #         tl.arange(0, CHUNK_SIZE)[:, None] * CHUNK_SIZE
    #         + tl.arange(0, CHUNK_SIZE)[None, :]
    #     )
    #     T = tl.load(T_ptr + T_group_offset + T_idx)
    # else:
    #     T = load_toeplitz(
    #         h_ptr,
    #         FILTER_LEN,
    #         CHUNK_SIZE,
    #         SINGLE_GROUP=SINGLE_GROUP,
    #         group_num=filter_group,
    #     )

    # T = tl.trans(T)

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
        # y2 = tl.load(y2_ptr + load_offsets)

        # Start backprop
        # dC = dy * y2
        # Backprop through C
        dy = dy * C

        # dy1 = tl.dot(
        #     T,
        #     dy,
        #     input_precision=input_precision,
        #     max_num_imprecise_acc=max_num_imprecise_acc,
        #     out_dtype=dy.dtype,
        # )
        # dx = dy1 * B
        # dB = dy1 * x

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
        # is_last_chunk = pid_chunk == chunks_per_seq - 1
        is_first_chunk = pid_chunk == 0

        # if not is_last_chunk:
        #     dy_lead_idx = dy_ptr + load_offsets + chunk_stride
        #     dy_lead = tl.load(dy_lead_idx)
        #     C_lead_idx = C_ptr + load_offsets + chunk_stride
        #     C_lead = tl.load(C_lead_idx)
        #     dy_lead *= C_lead

        #     if LOAD_T:
        #         # offset and idx defined above
        #         T_c = tl.load(T_hat_ptr + T_group_offset + T_idx)
        #     else:
        #         T_c = load_correction_toeplitz(
        #             h_ptr,
        #             FILTER_LEN,
        #             CHUNK_SIZE,
        #             SINGLE_GROUP=SINGLE_GROUP,
        #             group_num=filter_group,
        #         )
        #     T_c = tl.trans(T_c)

        #     dcorrection = tl.dot(
        #         T_c,
        #         dy_lead,
        #         input_precision=input_precision,
        #         max_num_imprecise_acc=max_num_imprecise_acc,
        #         out_dtype=dy.dtype,
        #     )
        #     dcorrection_dx = dcorrection * B
        #     dcorrection_dB = dcorrection * x
        #     dx += dcorrection_dx
        #     dB += dcorrection_dB

        # store_offsets = batch_offset + chunk_offset + col_offset + row_range + col_range

        # tl.store(dx_ptr + store_offsets, dx)
        # tl.store(dB_ptr + store_offsets, dB)
        # tl.store(dC_ptr + store_offsets, dC)

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
            B_lag_idx = B_ptr + load_offsets - chunk_stride
            B_lag = tl.load(B_lag_idx)
            x_lag_idx = x_ptr + load_offsets - chunk_stride
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
def _two_pass_bwd_grouped_kernel_dgrad(
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
    # Output ptrs
    dx_ptr,
    dB_ptr,
    dC_ptr,
    # Strides
    input_batch_stride,
    input_row_stride,
    input_col_stride,
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
    LOAD_BX: tl.constexpr = False,
    CHUNK_TILES_PER_PROGRAM: tl.constexpr = 1,
    ENABLE_CHECK: tl.constexpr = False,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
    # No need to set here since these can be passed directly to kernel call
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    # num_ctas: tl.constexpr = 1,
    DEBUG: tl.constexpr = True,
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
                "LOAD_BX:",
                LOAD_BX,
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

        # Correction term
        # In backwards, we roll in the opposite direction
        # Hence, the last chunk in the sequence does not need correction
        is_last_chunk = pid_chunk == chunks_per_seq - 1

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


def cgcg_conv1d_wgrad(Bx, C, dy, h, return_intermediates=False):
    """ """
    original_dtype = Bx.dtype
    bs, seqlen, g, dg = Bx.shape
    assert h.shape[0] == g
    hl = h.shape[-1]
    d = g * dg

    # Reshape Bx
    Bx = Bx.permute(0, 2, 3, 1)  # b, g, dg, l
    Bx = Bx.reshape(bs, -1, seqlen)  # b, d, l
    h_grouped = h.repeat_interleave(dg, dim=0)  # d, 1, hl

    grad_output = dy * C
    grad_channels_last = grad_output.permute(0, 2, 3, 1)
    grad_channels_last_flat = grad_channels_last.reshape(bs, -1, seqlen)

    stride = 1
    padding = hl - 1
    groups = d

    #
    # NOTE: conv1d raises CUDNN_BACKEND_OPERATION error if dtype == float16
    Bx = Bx.to(torch.float32)
    grad_channels_last_flat = grad_channels_last_flat.to(torch.float32)
    wgrad = F.grad.conv1d_weight(
        Bx, h_grouped.shape, grad_channels_last_flat, stride=stride, padding=padding, groups=groups
    )

    wgrad = wgrad.reshape(g, dg, 1, hl)
    h_grad = wgrad.sum(axis=1)

    if return_intermediates:
        return Bx.to(original_dtype), h_grouped.to(original_dtype), h_grad.to(original_dtype)

    return h_grad.to(original_dtype)


# TODO: use masking instead of lead / lag loading
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
    schedule: str = "default",
    autotune: bool = False,
    CHUNK_SIZE: int = None,
    BLOCK_D: int = None,
    NUM_PIPELINE_STAGES: int = 0,  # for tl.range
    THREADBLOCK_SWIZZLE: str = "row",
    CHUNK_TILES_PER_PROGRAM: int = 1,
    LOAD_BX: bool = False,
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
    # device_props = DeviceProps()

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
            kernel_dgrad: triton.runtime.JITFunction = _two_pass_bwd_grouped_kernel_dgrad
            kernel_wgrad: triton.runtime.JITFunction = _two_pass_bwd_grouped_kernel_wgrad
        else:
            raise ValueError(f"version {version} not implemented")

    if CHUNK_SIZE < filter_len:
        raise ValueError("CHUNK_SIZE must be >= filter_len")

    # basic shape checks
    assert dg >= 16, "dg must be >= 8 to use tensor-cores"
    assert x.shape == dy.shape == B.shape == C.shape == y2.shape
    assert hg == g
    assert _in_channel_div_group == 1

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
        assert T.shape == T_hat.shape == torch.Size([g, CHUNK_SIZE, CHUNK_SIZE])
        assert T.is_contiguous()
        assert T_hat.is_contiguous()
        # Kernel constexpr
        LOAD_T = True
    else:
        LOAD_T = False

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
        raise NotImplementedError("Persistent schedule not implemented")
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

    if version != "v3":
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
    else:
        if return_dgrad:
            kernel_args_dgrad = (
                dy,
                x,
                B,
                C,
                h,
                # Intermediate activations
                y2,
                T,
                T_hat,
                # Outputs
                dx,
                dB,
                dC,
                # Strides
                batch_stride,
                row_stride,
                col_stride,
                # Shapes
                bs,
                seqlen,
                g,
                dg,
            )

        if return_wgrad:
            kernel_args_wgrad = (
                dy,
                x,
                B,
                C,
                # Outputs
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
    # Inputs, saved from fwd

    kernel_constexprs = {
        "FILTER_LEN": filter_len,
        "SINGLE_GROUP": g == 1,
        "LOAD_T": LOAD_T,
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
        if version == "v1" or version == "v2":
            compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](*kernel_args, **kernel_constexprs)
        else:
            if return_dgrad:
                compiled_kernel_dgrad: triton.compiler.CompiledKernel = kernel_dgrad[grid](
                    *kernel_args_dgrad, **kernel_constexprs, LOAD_BX=LOAD_BX
                )

        dx = dx.reshape(bs, seqlen, g, dg)
        dB = dB.reshape(bs, seqlen, g, dg)
        dC = dC.reshape(bs, seqlen, g, dg)

        num_blocks_per_filter_group = dg // BLOCK_D

        if version == "v3":
            if return_dgrad:
                return dx, dB, dC

            if return_wgrad:
                # Run filter grad kernel
                compiled_kernel_wgrad = kernel_wgrad[grid](*kernel_args_wgrad, **kernel_constexprs)
        # Run second reduction pass
        dhdT = dhdT.reshape(bs, num_chunks, g, num_blocks_per_filter_group, filter_len, CHUNK_SIZE)
        dhdTc = dhdTc.reshape_as(dhdT)
        dhdT = dhdT.sum([0, 1, 3, 5]).reshape(*filter_shape)
        dhdTc = dhdTc.sum([0, 1, 3, 5]).reshape_as(dhdT)
        dh = dhdT + dhdTc

    if return_kernel:
        if version == "v3":
            compiled_kernel = compiled_kernel_dgrad, compiled_kernel_wgrad
        return dx, dB, dC, dh, compiled_kernel
    else:
        return dx, dB, dC, dh


if __name__ == "__main__":
    dtype = torch.float32  # torch.float16 leads to numerical differences
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    torch.manual_seed(0)

    # Shapes
    bs = 1
    seqlen = 1024
    hl = 128  # Filter size
    d = 4096
    g = 2
    dg = d // g
    # Only for debugging
    CHUNK_SIZE = max(hl, 32)  # seqlen // 4
    BLOCK_D = 32  # if dg > 32 else dg
    CHUNK_TILES_PER_PROGRAM = 1
    x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
    h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)
    dy = torch.randn_like(x)

    y2 = torch.randn_like(x)
    T = torch.randn(
        g,
        CHUNK_SIZE,
        CHUNK_SIZE,
        dtype=dtype,
        device=x.device,
    )
    T_hat = torch.randn_like(T)

    # y_ref = gcg_fwd_ref_debug(x, B, C, h, interleave=True)
    h_ = h.flip(-1)
    # T = toeplitz(h_[:, 0], chunk_size)
    # T_c = correction_toeplitz(h_[:, 0], chunk_size)

    # print(f"{target=} {NUM_SM=} {NUM_REGS=} {SIZE_SMEM=} {WARP_SIZE=}")
    num_warps = 4
    swizzle = "row"
    warmup = False
    autotune = False
    schedule = "default"
    return_kernel = False
    return_autotune_result = False
    #
    kernel_config_v1 = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "BLOCK_D": BLOCK_D,
        "NUM_PIPELINE_STAGES": 0,
        "THREADBLOCK_SWIZZLE": swizzle,
        "num_warps": num_warps,
        "num_stages": 2,
    }
    kernel_config_v2 = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "BLOCK_D": BLOCK_D,
        "CHUNK_TILES_PER_PROGRAM": CHUNK_TILES_PER_PROGRAM,
        "THREADBLOCK_SWIZZLE": swizzle,
        "num_warps": num_warps,
        "num_stages": 2,
        "DEBUG": False,
    }
    bwd_fn_v1 = lambda: two_pass_bwd_grouped(
        dy,
        x,
        B,
        C,
        h,
        y2,
        T=T,
        T_hat=T_hat,
        version="v1",
        schedule=schedule,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        NUM_PIPELINE_STAGES=0,
        CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
        num_warps=num_warps,
    )
    bwd_fn_v2 = lambda: two_pass_bwd_grouped(
        dy,
        x,
        B,
        C,
        h,
        y2,
        T=T,
        T_hat=T_hat,
        version="v2",
        schedule=schedule,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        NUM_PIPELINE_STAGES=0,
        CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
        num_warps=num_warps,
    )
    bwd_fn_v3 = lambda: two_pass_bwd_grouped(
        dy,
        x,
        B,
        C,
        h,
        y2,
        T=T,
        T_hat=T_hat,
        version="v3",
        schedule=schedule,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        NUM_PIPELINE_STAGES=0,
        CHUNK_TILES_PER_PROGRAM=CHUNK_TILES_PER_PROGRAM,
        num_warps=num_warps,
    )
    dx_v1, dB_v1, dC_v1, dh_v1 = bwd_fn_v1()
    dx_v2, dB_v2, dC_v2, dh_v2 = bwd_fn_v2()
    dx_v3, dB_v3, dC_v3, dh_v3 = bwd_fn_v3()

    dx_diff_v1v3 = (dx_v1 - dx_v3).abs().max()
    dx_diff_v1v2 = (dx_v1 - dx_v2).abs().max()
    print(f"dx_diff_v1v3={dx_diff_v1v3}")
    print(f"dx_diff_v1v2={dx_diff_v1v2}")
    from triton.testing import do_bench

    v1_t = do_bench(bwd_fn_v1)
    v2_t = do_bench(bwd_fn_v2)
    v3_t = do_bench(bwd_fn_v3)
    print(f"v1_t={v1_t}")
    print(f"v2_t={v2_t}")
    print(f"v3_t={v3_t}")
