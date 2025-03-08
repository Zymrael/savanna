import warnings

import torch

from savanna.kernels.triton_src.cgcg.src.bwd import two_pass_bwd_refactor
from savanna.kernels.triton_src.cgcg.src.fwd import two_pass_fwd_grouped
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    BwdKernelConfigRefactor,
    FwdKernelConfigRefactor,
)


def _unpack_kernel_config(fwd_kernel_cfg: FwdKernelConfigRefactor):
    if fwd_kernel_cfg is not None:
        BLOCK_D = fwd_kernel_cfg.BLOCK_D
        CHUNK_SIZE = fwd_kernel_cfg.CHUNK_SIZE
        THREADBLOCK_SWIZZLE = fwd_kernel_cfg.THREADBLOCK_SWIZZLE
        num_warps = fwd_kernel_cfg.num_warps
        num_stages = fwd_kernel_cfg.num_stages
        num_ctas = fwd_kernel_cfg.num_ctas
    else:
        BLOCK_D = CHUNK_SIZE = THREADBLOCK_SWIZZLE = num_warps = num_stages = num_ctas = None

    return BLOCK_D, CHUNK_SIZE, THREADBLOCK_SWIZZLE, num_warps, num_stages, num_ctas


class TwoPassChunkedGateConvGate(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h: torch.Tensor,
        bs: int,
        seqlen: int,
        g: int,
        dg: int,
        schedule: str = "default",
        fwd_autotune: bool = False,
        fwd_kernel_cfg: FwdKernelConfigRefactor = None,
        fused_bwd: bool = False,
        bwd_autotune: bool = False,
        bwd_kernel_cfg: BwdKernelConfigRefactor = None,
    ):
        schedule = schedule if fwd_autotune else fwd_kernel_cfg.schedule
        assert x.shape == torch.Size([bs, seqlen, g, dg])
        assert x.shape == B.shape == C.shape

        x = x if x.is_contiguous() else x.contiguous()
        B = B if B.is_contiguous() else B.contiguous()
        C = C if C.is_contiguous() else C.contiguous()
        h = h if h.is_contiguous() else h.contiguous()

        BLOCK_D, CHUNK_SIZE, THREADBLOCK_SWIZZLE, num_warps, num_stages, num_ctas = _unpack_kernel_config(
            fwd_kernel_cfg
        )
        out = two_pass_fwd_grouped(
            x,
            B,
            C,
            h,
            autotune=fwd_autotune,
            schedule=schedule,
            # Always return bx, y2 for backwards
            return_bx=True,
            return_y2=True,
            BLOCK_D=BLOCK_D,
            CHUNK_SIZE=CHUNK_SIZE,
            THREADBLOCK_SWIZZLE=THREADBLOCK_SWIZZLE,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
        )

        Bx, y2, y = out

        ctx.save_for_backward(x, B, C, h, Bx, y2)
        ctx.fwd_kernel_cfg = fwd_kernel_cfg
        ctx.input_shape = (bs, seqlen, g, dg)
        ctx.fused_bwd = fused_bwd
        ctx.bwd_autotune = bwd_autotune
        ctx.bwd_kernel_cfg = bwd_kernel_cfg
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, B, C, h, Bx, y2 = ctx.saved_tensors
        bs, seqlen, g, dg = ctx.input_shape

        dy = dy if dy.is_contiguous() else dy.contiguous()

        dx, dB, dC, dh = two_pass_bwd_refactor(
            dy=dy,
            Bx=Bx,
            y2=y2,
            x=x,
            B=B,
            C=C,
            h=h,
            bs=bs,
            seqlen=seqlen,
            g=g,
            dg=dg,
            fused=ctx.fused_bwd,
            autotune=ctx.bwd_autotune,
            kernel_config=ctx.bwd_kernel_cfg,
        )

        return (
            dx,
            dB,
            dC,
            dh,
            # Remaining are kwargs from fwd pass
            None,  # bs
            None,  # seqlen
            None,  # g
            None,  # dg
            None,  # schedule
            None,  # fwd_autotune
            None,  # FwdKernelConfig
            None,  # fused_bwd
            None,  # bwd_autotune
            None,  # BwdKernelConfig
        )


def two_pass_chunked_gate_conv_gate(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    bs: int,
    seqlen: int,
    g: int,
    dg: int,
    schedule: str = "default",  # @jeromeku: TODO: not used, remove
    # autotune only applies to fwd pass
    fwd_autotune: bool = False,
    bwd_autotune: bool = True,
    fused_bwd: bool = True,
    # manual kernel configs
    fwd_kernel_cfg: FwdKernelConfigRefactor = None,
    bwd_kernel_cfg: BwdKernelConfigRefactor = None,
):
    assert fwd_autotune or (fwd_kernel_cfg is not None), "Must specify fwd_kernel_cfg if not autotuning fwd"
    if fused_bwd:
        assert bwd_autotune or (
            bwd_kernel_cfg is not None
        ), "Must specify bwd_kernel_cfg if not autotuning bwd"
        if bwd_autotune and (bwd_kernel_cfg is not None):
            warnings.warn("WARNING: Both autotune and bwd_kernel_cfg specified, using bwd_kernel_cfg")
            bwd_autotune = False

    if fwd_autotune and (fwd_kernel_cfg is not None):
        warnings.warn("WARNING: Both autotune and fwd_kernel_cfg specified, using fwd_kernel_cfg")
        fwd_autotune = False

    return TwoPassChunkedGateConvGate.apply(
        x,
        B,
        C,
        h,
        bs,
        seqlen,
        g,
        dg,
        schedule,
        fwd_autotune,
        fwd_kernel_cfg,
        fused_bwd,
        bwd_autotune,
        bwd_kernel_cfg,
    )


# -- Previous implementation, disregard for now -- #
# class _TwoPassChunkedGateConvGate_old(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         x: torch.Tensor,
#         B: torch.Tensor,
#         C: torch.Tensor,
#         h: torch.Tensor,
#         schedule: str = "default",
#         autotune: bool = False,
#         fwd_kernel_cfg: FwdKernelConfig = None,
#         bwd_kernel_cfg: BwdKernelConfig = None,
#         # Always return y2 in fwd for backwards pass where y2 = T_local @ B*x + T_hat @ B_lag*x_lag
#         # return_y2: bool = True,
#     ):
#         return_toeplitz = bwd_kernel_cfg.LOAD_TOEPLITZ
#         return_bx_lag = bwd_kernel_cfg.LOAD_BX_LAG
#         schedule = schedule if autotune else fwd_kernel_cfg.schedule

#         # Chunk size for backwards must be the same as the forward chunk size if re-using toeplitz
#         if fwd_kernel_cfg is not None and return_toeplitz:
#             bwd_kernel_cfg.CHUNK_SIZE = fwd_kernel_cfg.CHUNK_SIZE
#         # TMA
#         if not autotune and fwd_kernel_cfg.USE_TMA:
#             raise NotImplementedError("Skip TMA for now")
#             # kernel = two_pass_fwd_grouped_tma
#         else:
#             kernel = two_pass_fwd_grouped

#         x = x if x.is_contiguous() else x.contiguous()
#         B = B if B.is_contiguous() else B.contiguous()
#         C = C if C.is_contiguous() else C.contiguous()
#         h = h if h.is_contiguous() else h.contiguous()

#         out = kernel(
#             x,
#             B,
#             C,
#             h,
#             autotune=autotune,
#             schedule=schedule,
#             return_toeplitz=return_toeplitz,
#             # Always return y2 in fwd for backwards
#             return_y2=True,
#             return_bx_lag=return_bx_lag,
#             CHUNK_SIZE=fwd_kernel_cfg.CHUNK_SIZE,
#             BLOCK_D=fwd_kernel_cfg.BLOCK_D,
#             NUM_PIPELINE_STAGES=fwd_kernel_cfg.NUM_PIPELINE_STAGES,
#             THREADBLOCK_SWIZZLE=fwd_kernel_cfg.THREADBLOCK_SWIZZLE,
#             num_warps=fwd_kernel_cfg.num_warps,
#             num_stages=fwd_kernel_cfg.num_stages,
#             num_ctas=fwd_kernel_cfg.num_ctas,
#             return_autotune_result=True if autotune else False,
#         )

#         # NOTE: T, T_hat, y2 are always returned, None if not return_toeplitz | return_toeplitz
#         # bwd handles None for T, T_hat
#         # Always return y2
#         if autotune:
#             y, T, T_hat, y2, autotuned_result = out
#             best_config: Config = autotuned_result.best_config
#             # Chunk size needs to be the same for fwd and bwd
#             CHUNK_SIZE = best_config.kwargs.get("CHUNK_SIZE")
#             bwd_kernel_cfg.CHUNK_SIZE = CHUNK_SIZE
#             # TODO: also set BLOCK_D, num_warps, NUM_PIPELINE_STAGES, etc. from autotuned_result?
#             # bwd_kernel_cfg.BLOCK_D = best_config.kwargs.get("BLOCK_D")
#             # bwd_kernel_cfg.num_warps = best_config.kwargs.get("num_warps")
#             # bwd_kernel_cfg.NUM_PIPELINE_STAGES = best_config.kwargs.get("NUM_PIPELINE_STAGES")
#             # bwd_kernel_cfg.THREADBLOCK_SWIZZLE = best_config.kwargs.get("THREADBLOCK_SWIZZLE")
#             # bwd_kernel_cfg.num_stages = best_config.kwargs.get("num_stages")
#             # bwd_kernel_cfg.num_ctas = best_config.kwargs.get("num_ctas")
#         else:
#             y, T, T_hat, y2, bx_lag = out

#         ctx.save_for_backward(x, B, C, h, T, T_hat, y2, bx_lag)
#         ctx.fwd_kernel_cfg = fwd_kernel_cfg
#         ctx.bwd_kernel_cfg = bwd_kernel_cfg

#         return y

#     @staticmethod
#     def backward(ctx, dy: torch.Tensor):
#         x, B, C, h, T, T_hat, y2, bx_lag = ctx.saved_tensors
#         dy = dy if dy.is_contiguous() else dy.contiguous()
#         x = x if x.is_contiguous() else x.contiguous()
#         B = B if B.is_contiguous() else B.contiguous()
#         C = C if C.is_contiguous() else C.contiguous()
#         h = h if h.is_contiguous() else h.contiguous()
#         # logger.debug(f"{dy=}\n{x=}\n{B=}\n{C=}\n{h=}\n{T=}\n{T_hat=}\n{y2=}")

#         kernel_config: BwdKernelConfig = ctx.bwd_kernel_cfg
#         # logger.debug(f"bwd kernel config: {kernel_config}")

#         if kernel_config.USE_TMA:
#             raise NotImplementedError("Skip TMA for now")
#             # kernel = two_pass_bwd_grouped_tma
#         else:
#             kernel = two_pass_bwd_grouped

#         dx, dB, dC, dh = kernel(
#             dy,
#             x,
#             B,
#             C,
#             h,
#             y2=y2,
#             T=T,
#             T_hat=T_hat,
#             bx_lag=bx_lag,
#             autotune=False,
#             version=kernel_config.version,
#             CHUNK_SIZE=kernel_config.CHUNK_SIZE,
#             BLOCK_D=kernel_config.BLOCK_D,
#             NUM_PIPELINE_STAGES=kernel_config.NUM_PIPELINE_STAGES,
#             THREADBLOCK_SWIZZLE=kernel_config.THREADBLOCK_SWIZZLE,
#             num_warps=kernel_config.num_warps,
#             num_stages=kernel_config.num_stages,
#             num_ctas=kernel_config.num_ctas,
#         )

#         return (
#             dx,
#             dB,
#             dC,
#             dh,
#             # Remaining are kwargs from fwd pass
#             None,  # autotune
#             None,  # schedule
#             None,  # FwdKernelConfig
#             None,  # BwdKernelConfig
#         )


# def _two_pass_chunked_gate_conv_gate(
#     x: torch.Tensor,
#     B: torch.Tensor,
#     C: torch.Tensor,
#     h: torch.Tensor,
#     return_toeplitz: bool = True,
#     # Only used if autotune
#     schedule: str = "default",
#     # autotune only applies to fwd pass
#     autotune: bool = False,
#     # manual kernel configs
#     fwd_kernel_cfg: FwdKernelConfig = None,
#     bwd_kernel_cfg: BwdKernelConfig = None,
# ):
#     assert autotune or (fwd_kernel_cfg is not None), "Must specify fwd_kernel_cfg if not autotuning"
#     if autotune and (fwd_kernel_cfg is not None):
#         warnings.warn("WARNING: Both autotune and fwd_kernel_cfg specified, using fwd_kernel_cfg")
#         autotune = False
#     assert bwd_kernel_cfg is not None, "Must specify bwd_kernel_cfg, autotune not supported for bwd pass"
#     if return_toeplitz:
#         # Check that fwd returns toeplitz and that bwd uses load_toeplitz
#         if fwd_kernel_cfg is not None:
#             fwd_kernel_cfg.RETURN_TOEPLITZ = True
#         bwd_kernel_cfg.LOAD_TOEPLITZ = True

#     return TwoPassChunkedGateConvGate.apply(
#         x,
#         B,
#         C,
#         h,
#         schedule,
#         autotune,
#         fwd_kernel_cfg,
#         bwd_kernel_cfg,
#     )
