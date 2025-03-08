# Copyright (c) 2024, Michael Poli, Eric Nguyen
import math
import os
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from savanna import mpu, print_rank_0
from savanna.logging import log_norm
from savanna.model.activations import get_activation
from savanna.model.operators.hyena.parametrization.explicit import (
    ExplicitSingleDecayFilter,
)
from savanna.model.operators.hyena.parametrization.implicit_complex import (
    ParallelComplexModalFilter,
)
from savanna.model.operators.hyena.parametrization.implicit_freeform import (
    ParallelImplicitFreeformFilter,
)
from savanna.model.operators.hyena.parametrization.implicit_modal import (
    ImplicitModalFilter,
    ImplicitRealModalFilter,
)
from savanna.ops.fftconv import _mul_sum, fftconv_func

try:
    from flashfftconv import FlashFFTConv
except ImportError:
    pass

try:
    from savanna.kernels.triton_src.cgcg.interface import (
        two_pass_chunked_gate_conv_gate,
    )
    from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
        BwdKernelConfigRefactor,
        FwdKernelConfigRefactor,
    )
    from savanna.kernels.triton_src.short_hyena.interface import run_short_hyena
    from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
        PostConvKernelConfig,
        PreConvKernelConfig,
        ShortHyenaOperatorKernelConfig,
    )

except ImportError:
    pass

try:
    from causal_conv1d import causal_conv1d_fn
except:
    causal_conv1d_fn = None
    print("no custom causal 1d conv installed, using default.")

from savanna.model.operators.hyena.distributed_a2a import AllToAllSingleFunction
from savanna.model.operators.hyena.p2p_cp_conv import (
    ExchangeOverlappingRegionsCausal,
    zigzag_get_overlapping_patches,
)


def initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with mpu.get_cuda_rng_tracker().fork():
        init_method(weight)


class ParallelCausalDepthwiseConv1d(nn.Module):
    def __init__(
        self,
        d_model,
        global_config,
        kernel_size,
        init_method,
        bias=False,  # not currently supported
        use_fast_causal_conv=False,
        num_groups=None,  # enables some weight sharing
        repeat_h_dg=True,
        local_init=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.use_bias = bias
        self.use_fast_causal_conv = use_fast_causal_conv
        self.num_groups = num_groups
        self.global_config = global_config
        self.use_cp_hyena = global_config.use_cp_hyena
        
        if self.num_groups is None:
            self.num_groups = self.d_model

        self.group_dim = self.d_model // self.num_groups

        if self.use_fast_causal_conv:
            assert causal_conv1d_fn is not None, "custom causal conv not installed"
            weight_shape = [self.num_groups, kernel_size]
        # use torch
        else:
            if global_config.use_depthwise_short_conv_grouping:
                weight_shape = [self.num_groups, 1, kernel_size]
                self.conv_groups = self.d_model

            else:
                if repeat_h_dg:
                    weight_shape = [self.num_groups, self.group_dim, kernel_size]
                else:
                    weight_shape = [self.num_groups, 1, kernel_size]

                self.conv_groups = self.num_groups

        self.short_conv_weight = nn.Parameter(
            torch.empty(
                weight_shape,
                device=torch.cuda.current_device(),
                dtype=global_config.params_dtype,
            )
        )

        # Use the standard PyTorch Conv1d class init: https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
        bounds = math.sqrt(1 / global_config.short_conv_L)
        conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
        if local_init:
            self.short_conv_weight.data = conv_init_method(self.short_conv_weight.data)
        else:
            initialize_affine_weight_gpu(self.short_conv_weight, conv_init_method, partition_dim=0)

    def forward(self, x):
        weight = self.short_conv_weight
        pad_size = self.kernel_size - 1

        # maybe handle num_groups
        weight = weight.repeat_interleave(self.group_dim, dim=0)

        if mpu.get_sequence_parallel_world_size() > 1 and self.use_cp_hyena:
            #print_rank_0(f"DEBUG::PARALLEL_CAUSAL_DEPTHWISE_CONV1D::forward::use_cp_hyena: {self.use_cp_hyena=}")
            cp_group = mpu.get_sequence_parallel_group()
            cp_rank = mpu.get_sequence_parallel_rank()

            # Transfer patches across ranks.
            seq_dim = 2  # Last dimension.
            chunk_a, chunk_b = zigzag_get_overlapping_patches(x, seq_dim=seq_dim, overlap_size=pad_size)
            received_a, received_b = ExchangeOverlappingRegionsCausal.apply(chunk_a, chunk_b, cp_group, cp_rank)

            # Pad and rearrange
            x = rearrange(x, "b h (nc s) -> (nc b) h s", nc=2)
            padding = torch.concat([received_a, received_b], dim=0)

            x = torch.concat([padding, x], dim=-1)
        else:
            x = F.pad(x, (pad_size, 0))

        if self.use_fast_causal_conv:
            # This function does additional pading under the hood. Unfortunately, there's no control over that.
            # Hence, we must do the de-padding manually.
            y = causal_conv1d_fn(x, weight, bias=None, activation=None)[..., pad_size:]

        else:
            L = x.shape[-1]

            y = F.conv1d(
                x,
                weight,
                bias=None,
                stride=1,
                padding=0,
                groups=self.conv_groups,
            )[..., :L]
    
        if mpu.get_sequence_parallel_world_size() > 1 and self.use_cp_hyena:
            y = rearrange(y,"(nc b) h s -> b h (nc s)", nc=2)

        return y


def get_groups_and_group_sizes(hidden_size, num_groups, world_size, expand_factor):
    width_per_tp_group = mpu.divide(hidden_size, world_size)
    num_groups_per_tp = int(mpu.divide(num_groups, world_size) * expand_factor)
    group_dim = width_per_tp_group // num_groups_per_tp
    return width_per_tp_group, num_groups_per_tp, group_dim


class ParallelHyenaOperator(nn.Module):

    def __init__(
        self,
        hidden_size,
        global_config,
        init_method,
        layer_number,
        downsample_factor=1,
        zigzag=True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.global_config = global_config
        self.layer_number = layer_number
        self.operator_type = global_config.operator_config[self.layer_number]
        self.fp16 = global_config.precision == "fp16"
        self.bf16 = global_config.precision == "bfloat16"
        self.cgcg_dtype = getattr(torch, global_config.cgcg_dtype)  # torch.float32
        self.act = get_activation(global_config, global_config.gating_act)

        if self.operator_type == "hyena_medium_conv" and global_config.hyena_medium_filter_cls is not None:
            self.hyena_filter_cls = global_config.hyena_medium_filter_cls
        else:
            self.hyena_filter_cls = global_config.hyena_filter_cls

        self.downsample_factor = downsample_factor
        self.bidirectional = global_config.bidirectional
        self.use_hyena_filter = global_config.use_hyena_filter
        self.use_fast_heads = global_config.use_fast_heads
        self.use_slow_heads = global_config.use_slow_heads
        
        self.zigzag = zigzag

        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()

        self.L = global_config.seq_length

        self.use_cp_hyena = global_config.use_cp_hyena

        if self.operator_type == "hyena_medium_conv":
            self.num_groups = (
                global_config.num_groups_hyena_medium
                if global_config.num_groups_hyena_medium is not None
                else global_config.num_groups_hyena
            )
        elif self.operator_type == "hyena_short_conv":
            self.num_groups = (
                global_config.num_groups_hyena_short
                if global_config.num_groups_hyena_short is not None
                else global_config.num_groups_hyena
            )
        else:
            # default to the global num_groups_hyena
            self.num_groups = global_config.num_groups_hyena

        if self.num_groups is None:
            self.num_groups = global_config.hidden_size

        world_size = mpu.get_model_parallel_world_size()

        self.width_per_tp_group, self.num_groups, self.group_dim = get_groups_and_group_sizes(
            self.hidden_size, self.num_groups, world_size, global_config.hyena_width_expansion
        )

        self.short_conv_L = global_config.short_conv_L
        self.use_medium_hyena = True if self.operator_type == "hyena_medium_conv" else False
        self.hyena_medium_conv_len = global_config.hyena_medium_conv_len

        self.log_hyena_norms = global_config.log_hyena_norms
        self.use_long_conv1d = global_config.use_long_conv1d
        self.use_flashfft = global_config.use_flashfft
        self.use_cgcg = global_config.use_cgcg
        self.is_medium_cgcg = self.use_cgcg and self.use_medium_hyena
        if self.use_flashfft:
            self.fftconv_fn = FlashFFTConv(self.L, dtype=torch.float16 if self.fp16 else torch.bfloat16)

        if self.use_medium_hyena and self.use_cgcg:
            if os.environ.get("SAVANNA_DEBUG", "0") == "1":
                import pdb

                pdb.set_trace()
            self.cgcg_fn = two_pass_chunked_gate_conv_gate

            self.cgcg_fwd_config = FwdKernelConfigRefactor(
                CHUNK_SIZE=self.global_config.cgcg_medium_fwd_kernel_config_chunk_size,
                BLOCK_D=min(self.group_dim, self.global_config.cgcg_medium_fwd_kernel_config_block_d),
                CHUNK_TILES_PER_PROGRAM=self.global_config.cgcg_medium_fwd_kernel_config_chunk_tiles_per_program,
                THREADBLOCK_SWIZZLE=self.global_config.cgcg_medium_fwd_kernel_config_threadblock_swizzle,
                num_warps=self.global_config.cgcg_medium_fwd_kernel_config_num_warps,
                num_stages=self.global_config.cgcg_medium_fwd_kernel_config_num_stages,
            )

            self.cgcg_bwd_config = BwdKernelConfigRefactor(
                pre_conv_BLOCK_X=self.global_config.cgcg_bwd_kernel_config_pre_conv_block_x,
                pre_conv_BLOCK_Y=self.global_config.cgcg_bwd_kernel_config_pre_conv_block_y,
                pre_conv_num_warps=self.global_config.cgcg_bwd_kernel_config_pre_conv_num_warps,
                post_conv_BLOCK_X=self.global_config.cgcg_bwd_kernel_config_post_conv_block_x,
                post_conv_BLOCK_Y=self.global_config.cgcg_bwd_kernel_config_post_conv_block_y,
                post_conv_num_warps=self.global_config.cgcg_bwd_kernel_config_post_conv_num_warps,
            )

        if self.hyena_filter_cls == "implicit_freeform":
            self.filter = ParallelImplicitFreeformFilter(
                global_config,
                init_method,
                d_model=self.num_groups,
                emb_dim=global_config.hyena_filter_emb_dim,
                order=global_config.hyena_filter_order,
                num_inner_mlps=global_config.hyena_filter_num_inner_mlps,
                seq_len=self.L,
                w=global_config.hyena_filter_w,
                normalized=global_config.normalize_hyena_filters,
                omega_0=global_config.hyena_filter_omega_0,
            )
        elif self.hyena_filter_cls == "explicit_single_decay":
            self.filter = ExplicitSingleDecayFilter(
                d_model=self.num_groups,
                L_cache=self.L,
                decay_preset=global_config.explicit_filter_decay_preset,
                num_decay_repeats=global_config.explicit_filter_num_decay_repeats,
            )
        elif self.hyena_filter_cls == "implicit_real_modal":
            self.filter = ImplicitRealModalFilter(
                d_model=self.num_groups,
                L_cache=self.L,
                order=global_config.hyena_filter_order,
                residue_factors=global_config.modal_residue_factors,
                pole_factors=global_config.modal_pole_factors,
            )
        elif self.hyena_filter_cls == "implicit_modal":
            self.filter = ImplicitModalFilter(
                d_model=self.num_groups,
                L_cache=self.L,
                order=global_config.hyena_filter_order,
                gamma_min=global_config.modal_gamma_min,
                gamma_max=global_config.modal_gamma_max,
            )
        elif self.hyena_filter_cls == "implicit_complex_modal":
            self.filter = ParallelComplexModalFilter(
                d_model=self.num_groups * 2 if self.bidirectional else self.num_groups,
                order=global_config.hyena_filter_order,
            )

        else:
            raise ValueError(f"Unknown hyena filter class: {self.hyena_filter_cls}")

        if self.use_slow_heads:
            self.conv_bias = nn.Parameter(
                torch.empty(
                    self.num_groups,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
        else:
            self.conv_bias = nn.Parameter(
                torch.empty(
                    self.width_per_tp_group,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )

        self.conv_bias.model_parallel = True
        self.conv_bias.partition_dim = 0
        self.conv_bias.stride = 1

    def multihead_forward(self, q, k, v, h):
        batch_size = q.shape[0]
        group_dim = self.group_dim
        num_groups = self.num_groups

        L = v.shape[-1]
        fft_size = 2 * L
        kv = rearrange(k, "b (h d1) l -> b d1 1 h l", d1=group_dim) * rearrange(
            v, "b (h d2) l -> b 1 d2 h l", d2=group_dim
        )
        if self.use_flashfft:
            # treat mhfftconv as a large batched fftconv
            kv_reshape = kv.reshape(-1, num_groups, L)
            y = self.fftconv_fn(kv_reshape, h[0])
            y = y.view(batch_size, group_dim, group_dim, num_groups, L)
        else:
            kv_f = torch.fft.rfft(kv.to(torch.float32), n=fft_size) / fft_size
            h_f = torch.fft.rfft(h.to(torch.float32), n=fft_size)  # h L+1

            y = torch.fft.irfft(kv_f * h_f, n=fft_size, norm="forward")[..., :L]
        y = y.to(dtype=q.dtype)

        if torch.distributed.get_rank() == 0 and self.log_hyena_norms:
            with torch.no_grad():
                log_norm(
                    y[0, 0, 0, 0].real.norm(2, dim=-1),
                    f"hy_y_2norm_L_{self.layer_number}",
                    self.global_config.iteration,
                )

        out = y + kv * self.conv_bias.unsqueeze(-1)
        q = rearrange(q, "b (h d1) l -> b d1 1 h l", d1=group_dim)
        z = _mul_sum(out, q)
        z = rearrange(z, "b d2 h l -> b (h d2) l")

        if torch.distributed.get_rank() == 0 and self.log_hyena_norms:
            with torch.no_grad():
                log_norm(
                    z[0, 0, 0].real.norm(2, dim=-1),
                    f"hy_z_2norm_L_{self.layer_number}",
                    self.global_config.iteration,
                )

        z = z.to(v.dtype)
        return z

    def forward(self, x1, x2, v):
        """
        Note:
            Input shapes: bs, seq_length, (num_groups, group_size)
            Output shapes: bs, seq_length, num_groups, group_size
        """
        B, L, G, DG = x1.shape
        
        if mpu.get_sequence_parallel_world_size() > 1 and self.use_cp_hyena:
            cp_group = mpu.get_sequence_parallel_group()
        else:
            cp_group = None

        downsampled = self.downsample_factor > 1

        # Only permute if not medium cgcg
        if not self.is_medium_cgcg:
            x1 = rearrange(x1, "b l g dg -> b (g dg) l", g=self.num_groups, dg=self.group_dim)
            x2 = rearrange(x2, "b l g dg -> b (g dg) l", g=self.num_groups, dg=self.group_dim)
            v = rearrange(v, "b l g dg -> b (g dg) l", g=self.num_groups, dg=self.group_dim)

        x1, x2, v = x1[..., :L], x2[..., :L], v[..., :L]

        if self.downsample_factor > 1:
            raise ValueError("Not supported with context parallel")
            x1 = x1[..., :: self.downsample_factor]
            x2 = x2[..., :: self.downsample_factor]
            v = v[..., :: self.downsample_factor]
            L = L // self.downsample_factor

        # The kernel length must be adjusted in CP settings
        _L_kernel = L if cp_group is None else L * len(torch.distributed.get_process_group_ranks(cp_group))

        if self.use_medium_hyena:
            h = self.filter(min(self.hyena_medium_conv_len, _L_kernel))
        else:
            h = self.filter(_L_kernel)

        if type(h) == tuple:
            h = h[0]
            
        conv_bias = self.conv_bias
        local_size = None

        if cp_group is not None and len(torch.distributed.get_process_group_ranks(cp_group)) > 1:
            x1, x2, v = [AllToAllSingleFunction.apply(tensor, cp_group, "split_to_full", True) for tensor in [x1, x2, v] ]
            # the tensors are now split across channels, but have full length.
            # [ B, H // num_ranks, L]
            # print_rank_0(f"DEBUG::PARALLELHYENAOPERATOR::FORWARD::AllToAllSINGLE: {x1.shape=} {x2.shape=} {v.shape=} {h.shape=}")
            rank = torch.distributed.get_rank(cp_group)
            local_size = self.num_groups // mpu.get_sequence_parallel_world_size()

            if isinstance(self.filter, (ParallelImplicitFreeformFilter, ParallelComplexModalFilter, ImplicitModalFilter)):
                h = h[:, rank * local_size:(rank + 1) * local_size]
            elif isinstance(self.filter, ExplicitSingleDecayFilter):
                h = h[rank * local_size:(rank + 1) * local_size]
            else:
                raise ValueError(f"Kernels of type {self.filter.__class__} have not been verified with CP.")

            local_bias_size = self.width_per_tp_group // mpu.get_sequence_parallel_world_size()
            conv_bias = self.conv_bias[rank * local_bias_size:(rank + 1) * local_bias_size]

        if self.use_slow_heads:
            return self.multihead_forward(x1, x2, v, h)

        elif self.use_long_conv1d:
            h = h.repeat_interleave(self.group_dim, dim=-2)
            z = x2 * v

            z = (
                F.conv1d(z, h[:, None].flip(-1), padding=L - 1, groups=v.shape[1])[..., :L]
                + conv_bias.unsqueeze(-1) * z
            )
            z = z.to(v.dtype)
            z = x1 * z

        elif self.is_medium_cgcg:
            # TODO: if the conditions are met, we should not rearrange to l last in the first place
            # @jeromeku, done as of 2024-09-28 refactor (see above)
            # x1 = rearrange(x1, "b (d g) l -> b l g d", g=self.num_groups)
            # x2 = rearrange(x2, "b (d g) l -> b l g d", g=self.num_groups)
            # v = rearrange(v, "b (d g) l -> b l g d", g=self.num_groups)
            dtype = x1.dtype
            if os.environ.get("SAVANNA_DEBUG", "0") == "1":
                import pdb

                pdb.set_trace()
            # Mapping from x1, x2, and v -> kernel args
            # x1 is post-gate (C)
            # x2 is pre-gate (B)
            # v is x

            if self.cgcg_dtype != dtype:
                x = v.to(self.cgcg_dtype)
                B = x2.to(self.cgcg_dtype)
                C = x1.to(self.cgcg_dtype)
                h = h[:, None].to(self.cgcg_dtype)
            else:
                x = v
                B = x2
                C = x1
                h = h[:, None]

            bs, seqlen, g, dg = x.shape

            # @jeromeku: Refactor as of 2024-09-28
            # No more backward kernel config
            # default schedule is "default" as other schedules are not supported
            # fwd_kernel config is of class FwdKernelConfigRefactor
            # Explicitly pass in shape for internal checking
            z = self.cgcg_fn(
                x=x,  # x1.to(self.cgcg_dtype),
                B=B,  # x2.to(self.cgcg_dtype),
                C=C,  # v.to(self.cgcg_dtype),
                h=h,  # h[:, None].to(self.cgcg_dtype),  # g, 1, filter_l
                bs=bs,
                seqlen=seqlen,
                g=g,
                dg=dg,
                fwd_autotune=False,  # @jeromeku explicitly set to False for now
                bwd_autotune=self.global_config.cgcg_bwd_autotune,
                fused_bwd=self.global_config.cgcg_fused_bwd,
                fwd_kernel_cfg=self.cgcg_fwd_config,
                bwd_kernel_cfg=None if self.global_config.cgcg_bwd_autotune else self.cgcg_bwd_config,
            )
            z = z.reshape(bs, seqlen, g * dg)
            if self.cgcg_dtype != dtype:
                z = z.to(dtype)
            return z
        else:
            h = h.repeat_interleave(self.group_dim, dim=-2)

            if self.global_config.use_flashfft:
                # squeeze h dim (kernel), to get rid of leading 1 dim
                h = h.squeeze(0)
                z = self.fftconv_fn(v, h, x2, x1)
            else:
                z = x2 * v
                with torch.autocast("cuda"):
                    z = fftconv_func(
                        z.to(torch.float32),
                        h.to(torch.float32),
                        conv_bias,
                        None,
                        gelu=False,
                        bidirectional=self.bidirectional,
                    )
                    z = z.to(v.dtype)
                z = x1 * z

        if downsampled:
            raise ValueError("Not supported with context parallel")
            z = z.repeat_interleave(self.downsample_factor, dim=-1)

        if cp_group is not None and len(torch.distributed.get_process_group_ranks(cp_group)) > 1:
            z = AllToAllSingleFunction.apply(z, cp_group, "full_to_split", True)
           # print_rank_0(f"DEBUG::PARALLELHYENAOPERATOR::FORWARD::AllToAllSINGLE: {z.shape=}")
            
        return rearrange(z, "b (g dg) l -> b l (g dg)", g=G)


class ParallelShortHyenaOperator(nn.Module):
    def __init__(
        self,
        hidden_size,
        global_config,
        init_method,
        short_conv_class,
        use_fast_causal_conv=False,
        is_mlp=False,
        local_init=False,
    ):
        super().__init__()
        self.global_config = global_config
        self.is_mlp = is_mlp
        self.hidden_size = hidden_size
        self.cgcg_dtype = getattr(torch, global_config.cgcg_dtype)
        self.use_cgcg_mlp = global_config.use_cgcg_mlp and self.is_mlp
        self.use_cgcg_short = global_config.use_cgcg_short and not self.is_mlp
        self.use_custom_hyena_mlp_kernel = global_config.use_custom_hyena_mlp_kernel
        self.use_custom_hyena_short_kernel = global_config.use_custom_hyena_short_kernel
        self.use_fast_causal_conv = use_fast_causal_conv

        world_size = mpu.get_model_parallel_world_size() if not local_init else 1

        # assert, if using fast_conv_mixer, then the hyena_short_conv_len must be 3
        if use_fast_causal_conv:
            assert (
                global_config.hyena_short_conv_len <= 4
            ), "fast_conv_mixer requires hyena_short_conv_len <= 4"

        # for mlp type
        if is_mlp:
            # option to have a different kernel size for the short conv inside the mlp
            if global_config.hyena_mlp_len is not None:
                kernel_size = global_config.hyena_mlp_len
            else:
                kernel_size = global_config.hyena_short_conv_len

            # check for fast causal conv
            if global_config.fast_hyena_mlp_conv:
                assert global_config.hyena_mlp_len <= 4, "fast_hyena_mlp_conv requires hyena_mlp_len <= 4"
                use_fast_causal_conv = True

            self.pregate = global_config.hyena_mlp_pregate
            self.postgate = global_config.hyena_mlp_postgate

            self.num_groups = (
                global_config.num_groups_hyena_mlp
                if global_config.num_groups_hyena_mlp is not None
                else global_config.num_groups_hyena
            )

            if self.num_groups is None:
                self.num_groups = global_config.hidden_size

            self.num_groups = int(self.num_groups * global_config.hyena_mlp_expansion_factor)
        # handle mixer case
        else:
            kernel_size = global_config.hyena_short_conv_len
            self.pregate = global_config.hyena_short_conv_pregate
            self.postgate = global_config.hyena_short_conv_postgate
            self.num_groups = (
                global_config.num_groups_hyena_short
                if global_config.num_groups_hyena_short is not None
                else global_config.num_groups_hyena
            )
            if self.num_groups is None:
                self.num_groups = global_config.hidden_size

            self.num_groups = int(self.num_groups * global_config.hyena_width_expansion)

        self.width_per_tp_group, self.num_groups, self.group_dim = get_groups_and_group_sizes(
            self.hidden_size, self.num_groups, world_size, global_config.hyena_width_expansion
        )

        self.short_conv = short_conv_class(
            self.width_per_tp_group,
            global_config=global_config,
            kernel_size=kernel_size,
            init_method=init_method,
            bias=global_config.conv_proj_bias,
            use_fast_causal_conv=use_fast_causal_conv,
            num_groups=self.num_groups,
            repeat_h_dg=False,
            local_init=local_init,
        )

        self.kernel_fn, self.fwd_kernel_cfg, self.bwd_kernel_cfg = self.prepare_kernel_configs()

    def prepare_kernel_configs(self):
        if self.is_mlp and self.use_cgcg_mlp:

            kernel_fn = two_pass_chunked_gate_conv_gate
            fwd_kernel_cfg = FwdKernelConfigRefactor(
                CHUNK_SIZE=self.global_config.cgcg_short_fwd_kernel_config_chunk_size,
                BLOCK_D=min(self.group_dim, self.global_config.cgcg_short_fwd_kernel_config_block_d),
                CHUNK_TILES_PER_PROGRAM=self.global_config.cgcg_short_fwd_kernel_config_chunk_tiles_per_program,
                THREADBLOCK_SWIZZLE=self.global_config.cgcg_short_fwd_kernel_config_threadblock_swizzle,
                num_warps=self.global_config.cgcg_short_fwd_kernel_config_num_warps,
                num_stages=self.global_config.cgcg_short_fwd_kernel_config_num_stages,
            )
            bwd_kernel_cfg = BwdKernelConfigRefactor(
                pre_conv_BLOCK_X=self.global_config.cgcg_bwd_kernel_config_pre_conv_block_x,
                pre_conv_BLOCK_Y=self.global_config.cgcg_bwd_kernel_config_pre_conv_block_y,
                pre_conv_num_warps=self.global_config.cgcg_bwd_kernel_config_pre_conv_num_warps,
                post_conv_BLOCK_X=self.global_config.cgcg_bwd_kernel_config_post_conv_block_x,
                post_conv_BLOCK_Y=self.global_config.cgcg_bwd_kernel_config_post_conv_block_y,
                post_conv_num_warps=self.global_config.cgcg_bwd_kernel_config_post_conv_num_warps,
            )
            return kernel_fn, fwd_kernel_cfg, bwd_kernel_cfg
        elif not self.is_mlp and self.use_cgcg_short:

            kernel_fn = two_pass_chunked_gate_conv_gate
            fwd_kernel_cfg = FwdKernelConfigRefactor(
                CHUNK_SIZE=self.global_config.cgcg_short_fwd_kernel_config_chunk_size,
                BLOCK_D=min(self.group_dim, self.global_config.cgcg_short_fwd_kernel_config_block_d),
                CHUNK_TILES_PER_PROGRAM=self.global_config.cgcg_short_fwd_kernel_config_chunk_tiles_per_program,
                THREADBLOCK_SWIZZLE=self.global_config.cgcg_short_fwd_kernel_config_threadblock_swizzle,
                num_warps=self.global_config.cgcg_short_fwd_kernel_config_num_warps,
                num_stages=self.global_config.cgcg_short_fwd_kernel_config_num_stages,
            )
            bwd_kernel_cfg = BwdKernelConfigRefactor(
                pre_conv_BLOCK_X=self.global_config.cgcg_bwd_kernel_config_pre_conv_block_x,
                pre_conv_BLOCK_Y=self.global_config.cgcg_bwd_kernel_config_pre_conv_block_y,
                pre_conv_num_warps=self.global_config.cgcg_bwd_kernel_config_pre_conv_num_warps,
                post_conv_BLOCK_X=self.global_config.cgcg_bwd_kernel_config_post_conv_block_x,
                post_conv_BLOCK_Y=self.global_config.cgcg_bwd_kernel_config_post_conv_block_y,
                post_conv_num_warps=self.global_config.cgcg_bwd_kernel_config_post_conv_num_warps,
            )
            return kernel_fn, fwd_kernel_cfg, bwd_kernel_cfg

        elif self.is_mlp and self.use_custom_hyena_mlp_kernel:
            fn = run_short_hyena
            fwd_kernel_cfg = ShortHyenaOperatorKernelConfig(
                PreConvKernelConfig(
                    BLOCK_M=256,
                    BLOCK_N=256,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
                PostConvKernelConfig(
                    BLOCK_M=128,
                    BLOCK_N=128,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
            )
            bwd_kernel_cfg = ShortHyenaOperatorKernelConfig(
                PreConvKernelConfig(
                    BLOCK_M=256,
                    BLOCK_N=256,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
                PostConvKernelConfig(
                    BLOCK_M=128,
                    BLOCK_N=128,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
            )
            return fn, fwd_kernel_cfg, bwd_kernel_cfg

        elif not self.is_mlp and self.use_custom_hyena_short_kernel:
            fn = run_short_hyena
            fwd_kernel_cfg = ShortHyenaOperatorKernelConfig(
                PreConvKernelConfig(
                    BLOCK_M=256,
                    BLOCK_N=256,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
                PostConvKernelConfig(
                    BLOCK_M=128,
                    BLOCK_N=128,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
            )
            bwd_kernel_cfg = ShortHyenaOperatorKernelConfig(
                PreConvKernelConfig(
                    BLOCK_M=256,
                    BLOCK_N=256,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
                PostConvKernelConfig(
                    BLOCK_M=128,
                    BLOCK_N=128,
                    NUM_PIPELINE_STAGES=1,
                    num_warps=4,
                    num_ctas=1,
                ),
            )
            return fn, fwd_kernel_cfg, bwd_kernel_cfg
        else:
            return None, None, None

    def forward(self, x1, x2, v):
        """
        Note:
            Input shapes: bs, seq_length, (num_groups, group_size)
            Output shapes: bs, seq_length, num_groups, group_size
        """
        B, L, G, DG = x1.shape

        if self.use_custom_hyena_mlp_kernel or self.use_custom_hyena_short_kernel:
            z = self.kernel_fn(
                x1,
                x2,
                v,
                self.short_conv.short_conv_weight,
                repeat_interleave=True,
                use_causal_conv=self.use_fast_causal_conv,
                autotune=False,
                fwd_kernel_cfg=self.fwd_kernel_cfg,
                bwd_kernel_cfg=self.bwd_kernel_cfg,
            )
            return rearrange(z, "b l g dg -> b l (g dg)", g=G)

        elif self.use_cgcg_mlp or self.use_cgcg_short:
            dtype = x1.dtype
            if os.environ.get("SAVANNA_DEBUG", "0") == "1":
                import pdb

                pdb.set_trace()
            # @jeromeku: Refactor as of 2024-09-28
            # No more backward kernel config
            # default schedule is "default" as other schedules are not supported
            # fwd_kernel config is of class FwdKernelConfigRefactor
            # Explicitly pass in shape for internal checking

            # Mapping from x1, x2, and v -> kernel args
            # x1 is post-gate (C)
            # x2 is pre-gate (B)
            # v is x

            if self.cgcg_dtype != dtype:
                x = v.to(self.cgcg_dtype)
                B = x2.to(self.cgcg_dtype)
                C = x1.to(self.cgcg_dtype)
                h = self.short_conv.short_conv_weight.to(self.cgcg_dtype)  # g, 1, filter_l
            else:
                x = v
                B = x2
                C = x1
                h = self.short_conv.short_conv_weight  # g, 1, filter_l

            bs, seqlen, g, dg = x.shape

            z = self.kernel_fn(
                x,  # x1.to(self.cgcg_dtype),
                B,  # x2.to(self.cgcg_dtype),
                C,  # v.to(self.cgcg_dtype),
                h,  # g, 1, filter_l
                bs=bs,
                seqlen=seqlen,
                g=g,
                dg=dg,
                # Explicitly set fwd autotune to False for now
                fwd_autotune=False,
                bwd_autotune=self.global_config.cgcg_bwd_autotune,
                fused_bwd=self.global_config.cgcg_fused_bwd,
                fwd_kernel_cfg=self.fwd_kernel_cfg,
                bwd_kernel_cfg=None if self.global_config.cgcg_bwd_autotune else self.bwd_kernel_cfg,
            )
            out = rearrange(z, "b l g d -> b l (g d)")
            if self.cgcg_dtype != dtype:
                out = out.to(dtype)
            return out
        else:
            x1 = rearrange(x1, "b l g dg -> b (g dg) l")
            x2 = rearrange(x2, "b l g dg -> b (g dg) l")
            v = rearrange(v, "b l g dg -> b (g dg) l")
            x1, x2, v = x1[..., :L], x2[..., :L], v[..., :L]
            z = x2 * v if self.pregate else v
            z = self.short_conv(z)
            z = x1 * z if self.postgate else z
            return rearrange(z, "b d l -> b l d")