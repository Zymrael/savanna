import torch

from savanna.kernels.triton_src.short_hyena.src.bwd_kernels import (
    _post_conv_bwd_kernel,
    _post_conv_bwd_kernel_autotune,
    _pre_conv_bwd_kernel,
    _pre_conv_bwd_kernel_autotune,
)
from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
    PostConvKernelConfig,
    PreConvKernelConfig,
    # get_grid,
    get_SMS,
)

NUM_SMS = None


def pre_conv_bwd(
    dkv_permuted: torch.Tensor,
    dq_permuted: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    autotune=False,
    BLOCK_M=None,
    BLOCK_N=None,
    NUM_PIPELINE_STAGES=0,
    num_warps=4,
    overwrite=False,
    verbose=False,
    return_kernel=False,
):
    seqlen, bs, np, hn = k.shape
    in_shape = (bs, np * hn, seqlen)
    out_shape = k.shape

    assert dkv_permuted.is_contiguous()
    assert dkv_permuted.shape == torch.Size(in_shape)
    assert dq_permuted.is_contiguous()
    assert dq_permuted.shape == torch.Size(in_shape)

    M = bs * np * hn
    N = seqlen

    # grid_x, grid_y = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    # total_blocks = grid_x * grid_y

    # Ensure we only call cuda driver on first call
    global NUM_SMS

    if NUM_SMS is None:
        NUM_SMS = get_SMS()

    grid = (NUM_SMS, 1, 1)
    # num_programs = min(NUM_SMS, total_blocks)
    # grid = (num_programs,)

    # if verbose:
    #     sm_estimate = (
    #         BLOCK_M * BLOCK_N * k.element_size() * max(1, NUM_PIPELINE_STAGES) * 2
    #     )
    #     print(f"{NUM_SMS=}, {total_blocks=}, {num_programs=} {sm_estimate=}")

    # if overwrite:
    #     kv = v
    # else:
    # kv = torch.empty(*out_shape, dtype=v.dtype, device=v.device)

    dk = torch.empty(*out_shape, dtype=dkv_permuted.dtype, device=dkv_permuted.device)
    dv = torch.empty(*out_shape, dtype=dkv_permuted.dtype, device=dkv_permuted.device)
    dq = torch.empty(*out_shape, dtype=dq_permuted.dtype, device=dq_permuted.device)
    if autotune:
        kernel = _pre_conv_bwd_kernel_autotune[grid](
            # Inputs from forward
            k_ptr=k,
            v_ptr=v,
            # Inputs from backwards
            dkv_permuted_ptr=dkv_permuted,
            dq_permuted_ptr=dq_permuted,
            # Outputs
            dk_ptr=dk,
            dv_ptr=dv,
            dq_ptr=dq,
            M=M,
            N=N,
        )
    else:
        kernel = _pre_conv_bwd_kernel[grid](
            # Inputs from forward
            k_ptr=k,
            v_ptr=v,
            # Inputs from backwards
            dq_permuted_ptr=dq_permuted,
            dkv_permuted_ptr=dkv_permuted,
            # Outputs
            dq_ptr=dq,
            dk_ptr=dk,
            dv_ptr=dv,
            M=M,
            N=N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
            num_warps=num_warps,
        )
    if return_kernel:
        return (dq, dk, dv), kernel
    return dq, dk, dv


def post_conv_bwd(
    dy: torch.Tensor,
    q_permuted: torch.Tensor,
    conv_out: torch.Tensor,
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
    # Permute from (bs, np, seqlen, hn) to (bs, np, hn, seqlen) then reshape to (bs, np * hn, seqlen)
    assert dy.ndim == 4
    assert dy.is_contiguous()

    bs, g, seqlen, dg = dy.shape
    final_shape = (bs, g * dg, seqlen)

    if not conv_out.is_contiguous():
        _, actual_seqlen, _ = conv_out.stride()
        assert actual_seqlen == seqlen + filter_len - 1
    else:
        actual_seqlen = seqlen

    BS = bs * g
    M = seqlen
    N = dg

    if BLOCK_N is not None:
        BLOCK_N = min(BLOCK_N, dg)
        #    out_shape = (BS, N, M)

        assert M % BLOCK_M == 0
        assert N % BLOCK_N == 0

    if verbose:
        print(f"BS: {BS}, M: {M}, N: {N}")

    dq_permuted = torch.empty(*final_shape, dtype=dy.dtype, device=dy.device)
    dconv_out = torch.empty(*final_shape, dtype=dy.dtype, device=dy.device)

    global NUM_SMS

    if NUM_SMS is None:
        NUM_SMS = get_SMS()

    # grid = get_grid(BS, M, N, NUM_SMS=NUM_SMS, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    grid = (NUM_SMS, 1, 1)

    conv_batch_stride = actual_seqlen * N
    conv_row_stride = N
    dy_batch_stride = M * N  # (seqlen * hn)
    dy_row_stride = N

    if verbose:
        print(
            f"{conv_batch_stride=}, {conv_row_stride=}, {dy_batch_stride=}, {dy_row_stride=}"
        )
    if autotune:
        kernel = _post_conv_bwd_kernel_autotune[grid](
            dy_ptr=dy,
            q_permuted_ptr=q_permuted,
            conv_out_ptr=conv_out,
            dq_permuted_ptr=dq_permuted,
            dconv_out_ptr=dconv_out,
            bs=bs,
            g=g,
            dg=dg,
            seqlen=seqlen,
            actual_seqlen=actual_seqlen,  # (seqlen)
            dy_batch_stride=dy_batch_stride,
            dy_row_stride=dy_row_stride,
            conv_out_batch_stride=conv_batch_stride,
            conv_out_row_stride=conv_row_stride,
        )
    else:
        kernel = _post_conv_bwd_kernel[grid](
            dy_ptr=dy,
            q_permuted_ptr=q_permuted,
            conv_out_ptr=conv_out,
            dq_permuted_ptr=dq_permuted,
            dconv_out_ptr=dconv_out,
            bs=bs,
            g=g,
            dg=dg,
            seqlen=seqlen,
            actual_seqlen=actual_seqlen,  # (seqlen)
            dy_batch_stride=dy_batch_stride,
            dy_row_stride=dy_row_stride,
            conv_out_batch_stride=conv_batch_stride,
            conv_out_row_stride=conv_row_stride,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
            num_warps=num_warps,
        )

    if return_kernel:
        return (dq_permuted, dconv_out), kernel

    return dq_permuted, dconv_out


def hyena_mlp_bwd(
    dy: torch.Tensor,
    # saved from forwards
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    q_permuted: torch.Tensor,
    kv_permuted: torch.Tensor,
    conv_out: torch.Tensor,
    autotune: bool = False,
    pre_conv_kernel_config: PreConvKernelConfig = None,
    post_conv_kernel_config: PostConvKernelConfig = None,
    use_causal_conv=False,
    overwrite=False,
    verbose=False,
    return_kernel=False,
    debug=False,
):
    # first call post_conv_backwards to get dconv_out and dq_permuted
    seqlen, bs, np, hn = k.shape
    filter_len = w.shape[-1]
    assert dy.is_contiguous()
    assert dy.shape == torch.Size((bs, np, seqlen, hn))
    assert q_permuted.shape == conv_out.shape == torch.Size((bs, np * hn, seqlen))

    # Backprop through post-gate
    dq_permuted, dconv_out = post_conv_bwd(
        dy=dy,
        q_permuted=q_permuted,
        conv_out=conv_out,
        filter_len=filter_len,
        autotune=autotune,
        BLOCK_M=post_conv_kernel_config.BLOCK_M
        if post_conv_kernel_config is not None
        else None,
        BLOCK_N=post_conv_kernel_config.BLOCK_N
        if post_conv_kernel_config is not None
        else None,
        NUM_PIPELINE_STAGES=post_conv_kernel_config.NUM_PIPELINE_STAGES
        if post_conv_kernel_config is not None
        else None,
        num_warps=post_conv_kernel_config.num_warps
        if post_conv_kernel_config is not None
        else None,
        # overwrite=overwrite,
        # verbose=verbose,
        # return_kernel=return_kernel,
    )

    # dgrad and wgrad
    if use_causal_conv:
        assert filter_len <= 4, "causal conv only supports filter len <= 4"
        dkv_permuted, dw = torch.autograd.grad(
            [conv_out], [kv_permuted, w], [dconv_out]
        )
    else:
        dkv_permuted, dw = torch.autograd.grad(
            [conv_out],
            [kv_permuted, w],
            grad_outputs=[dconv_out],
        )

    # Backprop through pre-gate
    dq, dk, dv = pre_conv_bwd(
        k=k,
        v=v,
        dq_permuted=dq_permuted,
        dkv_permuted=dkv_permuted,
        autotune=autotune,
        BLOCK_M=pre_conv_kernel_config.BLOCK_M
        if pre_conv_kernel_config is not None
        else None,
        BLOCK_N=pre_conv_kernel_config.BLOCK_N
        if pre_conv_kernel_config is not None
        else None,
        NUM_PIPELINE_STAGES=pre_conv_kernel_config.NUM_PIPELINE_STAGES
        if pre_conv_kernel_config is not None
        else None,
        num_warps=pre_conv_kernel_config.num_warps
        if pre_conv_kernel_config is not None
        else None,
    )

    if debug:
        return dconv_out, dq_permuted, dkv_permuted, dq, dk, dv, dw
    return dq, dk, dv, dw
