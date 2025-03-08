import torch

from savanna.kernels.triton_src.short_hyena.bwd import hyena_mlp_bwd
from savanna.kernels.triton_src.short_hyena.fwd import hyena_mlp_fwd
from savanna.kernels.triton_src.short_hyena.src.kernel_utils import ShortHyenaOperatorKernelConfig


class ShortHyenaOperator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        repeat_interleave: bool = True,
        use_causal_conv: bool = False,
        autotune: bool = False,
        fwd_kernel_cfg: ShortHyenaOperatorKernelConfig = None,
        bwd_kernel_cfg: ShortHyenaOperatorKernelConfig = None,
    ):
        if not autotune:
            assert (
                fwd_kernel_cfg is not None
            ), "Must provide fwd kernel config if not autotuning"
            assert (
                bwd_kernel_cfg is not None
            ), "Must provide bwd kernel config if not autotuning"
     
        hl = w.shape[-1]
        if use_causal_conv:
            assert hl <= 4, "causal conv only works with filter_len <= 4"
        with torch.enable_grad():
            q_permuted, kv_permuted, conv_out, y = hyena_mlp_fwd(
                q=q,
                k=k,
                v=v,
                w=w,
                repeat_interleave=repeat_interleave,
                autotune=autotune,
                pre_conv_kernel_config=fwd_kernel_cfg.pre_conv_kernel_config if not autotune else None,
                post_conv_kernel_config=fwd_kernel_cfg.post_conv_kernel_config if not autotune else None,
                use_causal_conv=use_causal_conv,
                return_intermediates=True,
            )
        ctx.save_for_backward(k, v, w, q_permuted, kv_permuted, conv_out)
        ctx.fwd_kernel_cfg = fwd_kernel_cfg
        ctx.bwd_kernel_cfg = bwd_kernel_cfg
        ctx.use_causal_conv = use_causal_conv
        ctx.autotune = autotune
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        k, v, w, q_permuted, kv_permuted, conv_out = ctx.saved_tensors
        autotune = ctx.autotune
        seqlen, bs, g, dg = k.shape
        hl = w.shape[-1]
        
        use_causal_conv = ctx.use_causal_conv
        pre_conv_kernel_config = None if autotune else ctx.bwd_kernel_cfg.pre_conv_kernel_config
        post_conv_kernel_config = None if autotune else ctx.bwd_kernel_cfg.post_conv_kernel_config

        dq, dk, dv, dw = hyena_mlp_bwd(
            dy=dy,
            k=k,
            v=v,
            w=w,
            q_permuted=q_permuted,
            kv_permuted=kv_permuted,
            conv_out=conv_out,
            use_causal_conv=use_causal_conv,
            autotune=autotune,
            pre_conv_kernel_config=pre_conv_kernel_config,
            post_conv_kernel_config=post_conv_kernel_config ,
        )
        # dw = dw_g.reshape(g, dg, 1, hl).sum(1)
        
        return (
            dq,
            dk,
            dv,
            dw,
            None, # repeat_interleave
            None,  # use_causal_conv
            None,  # autotune
            None,  # fwd_kernel_cfg
            None,  # bwd_kernel_cfg
        )


def run_short_hyena(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    repeat_interleave: bool = True,
    # w_g: torch.Tensor,
    use_causal_conv: bool = False,
    autotune: bool = False,
    fwd_kernel_cfg: ShortHyenaOperatorKernelConfig = None,
    bwd_kernel_cfg: ShortHyenaOperatorKernelConfig = None,
):
    assert autotune or (
        fwd_kernel_cfg is not None and bwd_kernel_cfg is not None
    ), "Must specify fwd_kernel_cfg and bwd_kernel_cfg if not autotuning"

    return ShortHyenaOperator.apply(
        q,
        k,
        v,
        w,
        repeat_interleave,
        # w_g, 
        use_causal_conv,
        autotune,
        fwd_kernel_cfg,
        bwd_kernel_cfg,
    )
