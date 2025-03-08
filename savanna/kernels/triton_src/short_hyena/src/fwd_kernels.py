
import triton
import triton.language as tl
from triton.runtime import autotune, heuristics

from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
    get_autotune_configs,
    get_debug_autotune_configs,
    get_dg_heuristic_fwd,
)


@triton.jit
def _pre_conv_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_permuted_ptr,
    q_permuted_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
    PERMUTE_Q: tl.constexpr = False,
):
    # pid_x = tl.program_id(0)
    # pid_y = tl.program_id(1)
    grid_x = tl.cdiv(M, BLOCK_M)
    grid_y = tl.cdiv(N, BLOCK_N)
    num_blocks = grid_x * grid_y
    num_programs = tl.num_programs(0)

    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)

    start_pid = tl.program_id(0)
    sm_id = start_pid

    for pid in tl.range(
        start_pid, num_blocks, num_programs, num_stages=NUM_PIPELINE_STAGES
    ):
        pid_y = pid % grid_y
        pid_x = (pid // grid_y) % grid_x
        # tl.static_print("sm_id", sm_id, "pid", pid, "pid_x", pid_x, "pid_y", pid_y)
        row_offsets = pid_x * BLOCK_M + row_idx
        col_offsets = pid_y * BLOCK_N + col_idx
        # tl.static_print("pid_x", pid_x, "pid_y", pid_y, "row_idx", row_idx, "col_idx", col_idx, "row_offsets", row_offsets, "col_offsets", col_offsets)
        rm = row_offsets * N
        rn = col_offsets
        k = tl.load(k_ptr + rm[:, None] + rn[None, :])
        v = tl.load(v_ptr + rm[:, None] + rn[None, :])
        if PERMUTE_Q:
            q = tl.load(q_ptr + rm[:, None] + rn[None, :])
            q = tl.trans(q)
        # tl.static_print("x", x)
        kv = tl.trans(k * v)
        # tl.static_print("x_t", x_t)
        kvn = row_offsets
        kvm = col_offsets * M
        # out_offsets = om[:, None] + on[None, :]
        # tl.static_print("out_offsets", out_offsets)
        tl.store(kv_permuted_ptr + kvm[:, None] + kvn[None, :], kv)
        if PERMUTE_Q:
            tl.store(q_permuted_ptr + kvm[:, None] + kvn[None, :], q)


@triton.jit
def _post_conv_fwd_kernel(
    conv_out_ptr,
    q_permuted_ptr,
    y_ptr,
    bs,
    g,
    dg,
    seqlen,
    # BS,
    # M,
    # N,
    input_batch_stride,
    input_row_stride,
    output_batch_stride,
    output_row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
):
    # pid_x = tl.program_id(0)
    # pid_y = tl.program_id(1)
    BS = bs * g
    M = dg
    N = seqlen

    grid_x = tl.cdiv(M, BLOCK_M)
    grid_y = tl.cdiv(N, BLOCK_N)
    num_blocks = grid_x * grid_y
    num_programs = tl.num_programs(0)
    #    batch_stride = M * N
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)

    start_pid = tl.program_id(0)
    # sm_id = start_pid

    for bid in range(0, BS):
        input_batch_offset = bid * input_batch_stride

        for pid in tl.range(
            start_pid, num_blocks, num_programs, num_stages=NUM_PIPELINE_STAGES
        ):
            pid_y = pid % grid_y
            pid_x = (pid // grid_y) % grid_x

            row_offsets = pid_x * BLOCK_M + row_idx
            col_offsets = pid_y * BLOCK_N + col_idx
            # tl.static_print(
            #     "bid", bid, "sm_id", sm_id, "pid", pid, "pid_x", pid_x, "pid_y", pid_y
            # )

            # rm = row_offsets * N
            # rm = row_offsets * input_row_stride
            # rn = col_offsets
            #   x_idx = input_batch_offset + rm[:, None] + rn[None, :]
            # NOTE: strides for conv and q are different due to conv_out being the output of a sliced conv operation
            cm = row_offsets * input_row_stride
            cn = col_offsets
            conv_idx = input_batch_offset + cm[:, None] + cn[None, :]

            q_batch_offset = bid * M * N
            qm = row_offsets * N
            qn = col_offsets
            q_idx = q_batch_offset + qm[:, None] + qn[None, :]

            conv_out = tl.load(conv_out_ptr + conv_idx)
            q_permuted = tl.load(q_permuted_ptr + q_idx)
            out = conv_out * q_permuted
            out = tl.trans(out)

            on = row_offsets
            om = col_offsets * M
            output_batch_offset = bid * output_batch_stride

            y_idx = output_batch_offset + om[:, None] + on[None, :]
            tl.store(y_ptr + y_idx, out)


_pre_conv_fwd_kernel_autotune = triton.autotune(
    configs=get_autotune_configs(), key=["M", "N"]
)(_pre_conv_fwd_kernel)
_pre_conv_fwd_kernel_debug_autotune = triton.autotune(
    configs=get_debug_autotune_configs(), key=["M", "N"]
)(_pre_conv_fwd_kernel)

_post_conv_fwd_kernel_autotune = autotune(configs=get_autotune_configs(), key=["bs", "g", "dg", "seqlen"])(heuristics(get_dg_heuristic_fwd())(_post_conv_fwd_kernel))


