import triton
import triton.language as tl
from triton.runtime import autotune, heuristics

from savanna.kernels.triton_src.short_hyena.src.kernel_utils import get_autotune_configs, get_dg_heuristic_bwd

NUM_SMS = None


# Backwards 1st pass
# permute dy
# Use the transposed indices to load q_permuted and conv_out
# multiply by q_permuted and conv_out to get dq_permuted and dconv_out
# write out to transposed indices
@triton.jit
def _post_conv_bwd_kernel(
    dy_ptr,
    q_permuted_ptr,
    conv_out_ptr,
    dq_permuted_ptr,
    dconv_out_ptr,
    bs,
    g,
    dg,
    seqlen,
    actual_seqlen,
    dy_batch_stride,
    dy_row_stride,
    conv_out_batch_stride,
    conv_out_row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
):
    # pid_x = tl.program_id(0)
    # pid_y = tl.program_id(1)

    BS = bs * g
    M = seqlen
    M_conv = actual_seqlen
    N = dg

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
        input_batch_offset = bid * dy_batch_stride

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
            rm = row_offsets * dy_row_stride
            rn = col_offsets
            dy_idx = input_batch_offset + rm[:, None] + rn[None, :]

            # q_batch_offset = bid * M * N
            # qm = row_offsets * N
            # qn = col_offsets
            # q_idx = q_batch_offset + qm[:, None] + qn[None, :]

            dy = tl.load(dy_ptr + dy_idx)
            # q_permuted = tl.load(q_permuted_ptr + q_idx)
            # out = conv_out * q_permuted
            dy_permuted = tl.trans(dy)

            on = row_offsets
            om_q = col_offsets * M
            om_conv = col_offsets * M_conv
            q_batch_offset = (
                bid * dy_batch_stride
            )  # Use q_offsets to load and store q and conv_out, use conv_batch_offset to load conv
            conv_batch_offset = bid * conv_out_batch_stride

            output_idx = q_batch_offset + om_q[:, None] + on[None, :]
            conv_load_idx = conv_batch_offset + om_conv[:, None] + on[None, :]
            q_permuted = tl.load(q_permuted_ptr + output_idx)
            conv_out = tl.load(conv_out_ptr + conv_load_idx)
            dq_permuted = dy_permuted * conv_out
            dconv_out = dy_permuted * q_permuted
            tl.store(dconv_out_ptr + output_idx, dconv_out)
            tl.store(dq_permuted_ptr + output_idx, dq_permuted)

            # tl.store(x_permuted_ptr + x_permuted_idx, x_permuted)


# Load dkv, dq in permuted shape
# Transpose
# Load k and v using transposed index
# Calculate dk and dv
# Store in permuted shape
@triton.jit
def _pre_conv_bwd_kernel(
    # Inputs from forward pass
    k_ptr,
    v_ptr,
    # Inputs from backward pass
    dq_permuted_ptr,
    dkv_permuted_ptr,
    # Outputs
    dq_ptr,
    dk_ptr,
    dv_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr = 0,
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
    # sm_id = start_pid

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
        # k = tl.load(k_ptr + rm[:, None] + rn[None, :])
        # v = tl.load(v_ptr + rm[:, None] + rn[None, :])
        # if PERMUTE_Q:
        dq_permuted = tl.load(dq_permuted_ptr + rm[:, None] + rn[None, :])
        dkv_permuted = tl.load(dkv_permuted_ptr + rm[:, None] + rn[None, :])

        dq = tl.trans(dq_permuted)
        dkv = tl.trans(dkv_permuted)

        #
        # kv = tl.trans(k * v)
        # tl.static_print("x_t", x_t)
        kvn = row_offsets
        kvm = col_offsets * M
        kv_load_idx = kvm[:, None] + kvn[None, :]
        k = tl.load(k_ptr + kv_load_idx)
        v = tl.load(v_ptr + kv_load_idx)
        dk = dkv * v
        dv = dkv * k
        # tl.store(kv_ptr + kvm[:, None] + kvn[None, :], kv)
        # if PERMUTE_Q:
        tl.store(dq_ptr + kv_load_idx, dq)
        tl.store(dk_ptr + kv_load_idx, dk)
        tl.store(dv_ptr + kv_load_idx, dv)


_pre_conv_bwd_kernel_autotune = autotune(
    key=["M", "N"], configs=get_autotune_configs()
)(_pre_conv_bwd_kernel)

_post_conv_bwd_kernel_autotune = autotune(
    key=["bs", "g", "dg", "seqlen", "actual_seqlen"], configs=get_autotune_configs()
)(heuristics(get_dg_heuristic_bwd())(_post_conv_bwd_kernel))
