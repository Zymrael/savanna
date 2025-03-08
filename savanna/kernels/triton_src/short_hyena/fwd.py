import torch

from savanna.kernels.triton_src.short_hyena.src.fwd_kernels import (
    _post_conv_fwd_kernel,
    _post_conv_fwd_kernel_autotune,
    _pre_conv_fwd_kernel,
    _pre_conv_fwd_kernel_autotune,
    _pre_conv_fwd_kernel_debug_autotune,
)
from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
    PostConvKernelConfig,
    PreConvKernelConfig,
    get_SMS,
)
from savanna.kernels.triton_src.short_hyena.utils import causal_conv, torch_conv

NUM_SMS = None


def pre_conv_fwd(
    q,
    k,
    v,
    autotune=False,
    BLOCK_M=None,
    BLOCK_N=None,
    NUM_PIPELINE_STAGES=0,
    num_warps=4,
    overwrite=False,
    verbose=False,
    return_kernel=False,
    debug=False,
):
    seqlen, bs, g, dg = k.shape
    out_shape = [bs, g * dg, seqlen]

    # Logical M and N for the kernel
    M = seqlen
    N = bs * g * dg

    # grid_x, grid_y = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    # total_blocks = grid_x * grid_y

    # # Ensure we only call cuda driver on first call
    global NUM_SMS
    if NUM_SMS is None:
        NUM_SMS = get_SMS()

    # num_programs = min(NUM_SMS, total_blocks)
    grid = (NUM_SMS,)

    # if verbose:
    #     sm_estimate = (
    #         BLOCK_M * BLOCK_N * k.element_size() * max(1, NUM_PIPELINE_STAGES) * 2
    #     )
    #     print(f"{NUM_SMS=}, {total_blocks=}, {num_programs=} {sm_estimate=}")

    if overwrite:
        kv_permuted = v
    else:
        kv_permuted = torch.empty(
            *out_shape, dtype=v.dtype, device=v.device, requires_grad=True
        )

    q_permuted = torch.empty(
        *out_shape, dtype=q.dtype, device=q.device, requires_grad=True
    )

    if autotune:
        if debug:
            kernel = _pre_conv_fwd_kernel_debug_autotune[grid](
                q_ptr=q,
                k_ptr=k,
                v_ptr=v,
                q_permuted_ptr=q_permuted,
                kv_permuted_ptr=kv_permuted,
                M=M,
                N=N,
                PERMUTE_Q=True,
            )
        else:
            kernel = _pre_conv_fwd_kernel_autotune[grid](
                q_ptr=q,
                k_ptr=k,
                v_ptr=v,
                q_permuted_ptr=q_permuted,
                kv_permuted_ptr=kv_permuted,
                M=M,
                N=N,
                PERMUTE_Q=True,
            )
    else:
        breakpoint()
        kernel = _pre_conv_fwd_kernel[grid](
            q_ptr=q,
            k_ptr=k,
            v_ptr=v,
            q_permuted_ptr=q_permuted,
            kv_permuted_ptr=kv_permuted,
            M=M,
            N=N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
            PERMUTE_Q=True,
            num_warps=num_warps,
        )

    if return_kernel:
        return kv_permuted, q_permuted, kernel
    return kv_permuted, q_permuted


def post_conv_fwd(
    conv_out: torch.Tensor,
    q_permuted: torch.Tensor,
    out_shape: tuple,
    filter_len: int,
    autotune=False,
    BLOCK_M=None,
    BLOCK_N=None,
    NUM_PIPELINE_STAGES=0,
    num_warps=4,
    overwrite=False,
    return_kernel=False,
    verbose=False,
):
    bs, g, seqlen, dg = out_shape
    assert conv_out.ndim == 3
    # After F.conv1d, we slice the output to get an effective sequence length that matches the original sequence length
    # However, this leads to incontiguity, since the actual storage contains the entire convolved sequence, which is
    # sequence length + filter length - 1.
    # Since triton expects contiguous tensors, we need to account for this
    if not conv_out.is_contiguous():
        _, actual_seq_len, _ = conv_out.stride()
        assert actual_seq_len == seqlen + filter_len - 1
    else:
        actual_seq_len = seqlen

    BS = bs * g
    M = dg
    N = seqlen

    if BLOCK_M is not None:
        BLOCK_M = min(BLOCK_M, dg)

    # assert M % BLOCK_M == 0
    # assert N % BLOCK_N == 0
    # grid_m, grid_n = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    # assert grid_m * grid_n % NUM_SMS == 0
    # if verbose:
    #     print(f"x.shape: {conv_out.shape}, z.shape: {conv_out.shape}")
    #     print(f"BS: {BS}, M: {M}, N: {N}")

    if overwrite:
        out = conv_out
    else:
        out = torch.empty(
            *out_shape, dtype=conv_out.dtype, device=conv_out.device
        )  # (conv_out)

    global NUM_SMS

    if NUM_SMS is None:
        NUM_SMS = get_SMS()

    # grid = get_grid(BS, M, N, NUM_SMS=NUM_SMS, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    grid = (NUM_SMS, 1, 1)

    # if verbose:
    #     sm_estimate = (
    #         BLOCK_M
    #         * BLOCK_N
    #         * conv_out.element_size()
    #         * max(1, NUM_PIPELINE_STAGES)
    #         * 2
    #     )
    #     num_programs = grid[0]
    #     print(
    #         f"{NUM_SMS=}, {BS * grid_m * grid_n}, {grid_m * grid_n=}, {num_programs=}, {sm_estimate=}"
    #     )

    #    batch_stride, row_stride, _ = x.stride()
    # batch_stride = M * N
    # row_stride = N
    # NOTE the change in batch_stride and row_stride needed for incontiguity
    input_batch_stride = M * actual_seq_len
    input_row_stride = actual_seq_len
    output_batch_stride = M * N
    output_row_stride = N

    if autotune:
        kernel = _post_conv_fwd_kernel_autotune[grid](
            conv_out_ptr=conv_out,
            q_permuted_ptr=q_permuted,
            y_ptr=out,
            bs=bs,
            g=g,
            dg=dg,
            seqlen=seqlen,
            input_batch_stride=input_batch_stride,
            input_row_stride=input_row_stride,
            output_batch_stride=output_batch_stride,
            output_row_stride=output_row_stride,
        )
    else:
        kernel = _post_conv_fwd_kernel[grid](
            conv_out_ptr=conv_out,
            q_permuted_ptr=q_permuted,
            y_ptr=out,
            bs=bs,
            g=g,
            dg=dg,
            seqlen=seqlen,
            input_batch_stride=input_batch_stride,
            input_row_stride=input_row_stride,
            output_batch_stride=output_batch_stride,
            output_row_stride=output_row_stride,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
            num_warps=num_warps,
        )
    out = out.reshape(*out_shape)

    if return_kernel:
        return out, kernel

    return out


def hyena_mlp_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    repeat_interleave=True,
    # w_g: torch.Tensor,
    autotune=False,
    pre_conv_kernel_config: PreConvKernelConfig = None,
    post_conv_kernel_config: PostConvKernelConfig = None,
    use_causal_conv=False,
    overwrite=False,
    return_intermediates=False,
    verbose=False,
    debug=False,
):
    seqlen, bs, g, dg = q.shape
    out_shape = (bs, g, seqlen, dg)
    filter_len = w.shape[-1]

    if not autotune:
        assert (
            pre_conv_kernel_config is not None
        ), "Please specify a pre conv kernel config if not autotuning"
        assert (
            post_conv_kernel_config is not None
        ), "Please specify a post kernel config if not autotuning"
    # else:
    #     raise NotImplementedError("Autotuning not implemented yet")


    kv_permuted, q_permuted = pre_conv_fwd(
        q=q,
        k=k,
        v=v,
        autotune=autotune,
        BLOCK_M=pre_conv_kernel_config.BLOCK_M if not autotune else None,
        BLOCK_N=pre_conv_kernel_config.BLOCK_N if not autotune else None,
        NUM_PIPELINE_STAGES=pre_conv_kernel_config.NUM_PIPELINE_STAGES
        if not autotune
        else None,
        num_warps=pre_conv_kernel_config.num_warps if not autotune else None,
        overwrite=overwrite,
        debug=debug,
    )
  
    if repeat_interleave:
        assert w.shape[0] == g
        w = w.repeat_interleave(dg, dim=0)
        # w.retain_grad()
        # print(f"{w.shape=}")
        # assert w.shape[0] = g * dg

    if use_causal_conv:
        assert filter_len <= 4, "causal conv only works with filter_len <= 4"
        # conv_out = causal_conv(kv_permuted, w)
        conv_out = causal_conv(kv_permuted, w)
    else:
        # conv_out = causal_conv(kv_permuted, w)
        conv_out = torch_conv(kv_permuted, w)

    out = post_conv_fwd(
        conv_out=conv_out,
        q_permuted=q_permuted,
        out_shape=out_shape,
        filter_len=filter_len,
        autotune=autotune,
        BLOCK_M=post_conv_kernel_config.BLOCK_M if not autotune else None,
        BLOCK_N=post_conv_kernel_config.BLOCK_N if not autotune else None,
        NUM_PIPELINE_STAGES=post_conv_kernel_config.NUM_PIPELINE_STAGES
        if not autotune
        else None,
        num_warps=post_conv_kernel_config.num_warps if not autotune else None,
        overwrite=overwrite,
        verbose=verbose,
    )

    if return_intermediates:
        return q_permuted, kv_permuted, conv_out, out

    return out
