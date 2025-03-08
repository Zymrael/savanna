# Copyright (c) 2024, Michael Poli, Eric Nguyen

import math
import os
import sys
from functools import partial
from re import U

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from torch.nn.parameter import Parameter

from savanna import mpu
from savanna.logging import log_norm, tb_wandb_log
from savanna.model.activations import Sin, get_activation
from savanna.model.init_functions import xavier_normal_init_method
from savanna.model.operators.hyena.parametrization.explicit_filter import (
    ExplicitSingleDecayFilter,
    ImplicitRealModalFilter,
)
from savanna.model.operators.hyena.parametrization.implicit_complex import (
    ParallelComplexModalFilter,
)
from savanna.model.operators.hyena.parametrization.implicit_freeform import (
    ParallelImplicitFreeformFilter,
)
from savanna.model.operators.hyena.parametrization.implicit_utils import (
    PositionalEmbedding,
)
from savanna.model.operators.hyena.parametrization.utils import get_dtype_from_string
from savanna.ops.fftconv import _mul_sum, fftconv_func, fftconv_heads_ref

try:
    from flashfftconv import FlashFFTConv
except ImportError:
    pass

try:
    from savanna.kernels.triton_src.cgcg.interface import TwoPassChunkedGateConvGate
    from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
        BwdKernelConfig,
        FwdKernelConfig,
    )
except ImportError:
    pass

try:
    from causal_conv1d import causal_conv1d_fn
except:
    causal_conv1d_fn = None
    print("no custom causal 1d conv installed, using default.")


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
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.use_bias = bias
        self.use_fast_causal_conv = use_fast_causal_conv
        self.num_groups = num_groups
        
        if self.num_groups is None:
            self.num_groups = self.d_model
            
        self.group_dim = self.d_model // self.num_groups    

        if self.use_fast_causal_conv:
            assert causal_conv1d_fn is not None, "custom causal conv not installed!!!"
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
                    weight_shape = [self.num_groups, kernel_size]
                
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
        initialize_affine_weight_gpu(self.short_conv_weight, conv_init_method, partition_dim=0)


    def forward(self, x):
        weight = self.short_conv_weight
        
        # maybe handle num_groups
        weight = weight.repeat_interleave(self.group_dim, dim=0)

        if self.use_fast_causal_conv:
            y = causal_conv1d_fn(x, weight, bias=None, activation=None)

        else:
            L = x.shape[-1]

            y = F.conv1d(
                x,
                weight,
                bias=None,
                stride=1,
                padding=self.kernel_size - 1,
                groups=self.conv_groups,
            )[..., :L]
        return y
    

# class ParallelCausalDepthwiseConv1d(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         global_config,
#         kernel_size,
#         init_method,
#         bias=False,  # not currently supported
#         use_fast_causal_conv=False,
#         num_groups=None,  # enables some weight sharing
#         repeat_h_dg=True,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.kernel_size = kernel_size
#         self.use_bias = bias
#         self.use_fast_causal_conv = use_fast_causal_conv
#         self.num_groups = num_groups

#         if self.num_groups is None:
#             self.num_groups = self.d_model

#         self.group_dim = self.d_model // self.num_groups

#         if self.use_fast_causal_conv:
#             assert causal_conv1d_fn is not None, "custom causal conv not installed!!!"
#             weight_shape = [self.num_groups, kernel_size]
#         # use torch
#         else:
#             if repeat_h_dg:
#                 weight_shape = [self.num_groups, self.group_dim, kernel_size]
#             else:
#                 weight_shape = [self.num_groups, kernel_size]

#         self.short_conv_weight = nn.Parameter(
#             torch.empty(
#                 weight_shape,
#                 device=torch.cuda.current_device(),
#                 dtype=global_config.params_dtype,
#             )
#         )

#         # Use the standard PyTorch Conv1d class init: https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
#         bounds = math.sqrt(1 / global_config.short_conv_L)
#         conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
#         initialize_affine_weight_gpu(self.short_conv_weight, conv_init_method, partition_dim=0)

#     def forward(self, x):
#         weight = self.short_conv_weight

#         # maybe handle num_groups
#         weight = weight.repeat_interleave(self.group_dim, dim=0)

#         if self.use_fast_causal_conv:
#             y = causal_conv1d_fn(x, weight, bias=None, activation=None)

#         else:
#             L = x.shape[-1]

#             y = F.conv1d(
#                 x,
#                 weight,
#                 bias=None,
#                 stride=1,
#                 padding=self.kernel_size - 1,
#                 groups=self.num_groups,
#             )[..., :L]
#         return y


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
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.global_config = global_config
        self.layer_number = layer_number
        self.operator_type = global_config.operator_config[self.layer_number]
        self.fp16 = global_config.precision == "fp16"
        self.bf16 = global_config.precision == "bfloat16"
        self.cgcg_dtype = torch.float32
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

        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()

        self.L = global_config.seq_length

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
        if self.use_flashfft:
            self.fftconv_fn = FlashFFTConv(self.L, dtype=torch.float16 if self.fp16 else torch.bfloat16)

        if self.use_medium_hyena and self.use_cgcg:
            self.cgcg_fn = TwoPassChunkedGateConvGate

            self.cgcg_fwd_config = FwdKernelConfig(
                schedule="default",
                CHUNK_SIZE=128,
                BLOCK_D=min(self.group_dim, 128),
                THREADBLOCK_SWIZZLE="row",
                name="fwd",
                version="v2",
                RETURN_TOEPLITZ=True,
                RETURN_Y2=True,
                RETURN_BX_LAG=True,
                num_warps=8,
            )

            self.cgcg_bwd_config = BwdKernelConfig(
                schedule="default",
                CHUNK_SIZE=128,
                BLOCK_D=min(self.group_dim, 128),
                THREADBLOCK_SWIZZLE="row",
                name="bwd",
                version="v2",
                LOAD_TOEPLITZ=True,
                LOAD_BX_LAG=True,
                num_warps=8,
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
            )
        elif self.hyena_filter_cls == "implicit_real_modal":
            self.filter = ImplicitRealModalFilter(
                d_model=self.num_groups,
                L_cache=self.L,
                order=global_config.hyena_filter_order,
                residue_factors=global_config.modal_residue_factors,
                pole_factors=global_config.modal_pole_factors,
            )
        elif self.hyena_filter_cls == "implicit_complex_modal":
            self.filter = ParallelComplexModalFilter(
                H=self.num_groups * 2 if self.bidirectional else self.num_groups,
                N=global_config.hyena_filter_order,
            )

        else:
            raise ValueError(f"Unknown hyena filter class: {self.hyena_filter_cls}")

        if self.use_slow_heads:
            self.long_conv_bias = nn.Parameter(
                torch.empty(
                    self.num_groups,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
        else:
            self.long_conv_bias = nn.Parameter(
                torch.empty(
                    self.width_per_tp_group,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )

        self.long_conv_bias.model_parallel = True
        self.long_conv_bias.partition_dim = 0
        self.long_conv_bias.stride = 1

    def multihead_forward(self, q, k, v, h):
        batch_size = q.shape[0]
        group_dim = self.group_dim
        num_groups = self.num_groups  # used for flashfft
        L = v.shape[-1]
        fft_size = 2 * L
        kv = rearrange(k, "b (h d1) l -> b d1 1 h l", d1=group_dim) * rearrange(
            v, "b (h d2) l -> b 1 d2 h l", d2=group_dim
        )  # b d1 d2 h l
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

        # b d1 d2 h l
        out = y + kv * self.long_conv_bias.unsqueeze(-1)  # b d1 d2 h l
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

    def forward(self, query_layer, key_layer, value_layer):

        # input sizes: [sq, b, np, hn]
        # seqlen, batch, tensor parallel, hidden size per tensor parallel
        downsampled = self.downsample_factor > 1

        np = query_layer.shape[-2]
        L = query_layer.shape[0]
        query = rearrange(query_layer, "sq b np hn -> b (np hn) sq")
        key = rearrange(key_layer, "sq b np hn -> b (np hn) sq")
        value = rearrange(value_layer, "sq b np hn -> b (np hn) sq")

        q, k, v = query[..., :L], key[..., :L], value[..., :L]

        if self.downsample_factor > 1:
            q = q[..., :: self.downsample_factor]
            k = k[..., :: self.downsample_factor]
            v = v[..., :: self.downsample_factor]
            L = L // self.downsample_factor

        if self.use_medium_hyena:
            h = self.filter(min(self.hyena_medium_conv_len, L))
        else:
            h = self.filter(L)

        if type(h) == tuple:
            h = h[0]

        if self.use_slow_heads:
            return self.multihead_forward(q, k, v, h)

        elif self.use_long_conv1d:
            h = h.repeat_interleave(self.group_dim, dim=-2)
            z = k * v

            z = (
                F.conv1d(z, h[:, None].flip(-1), padding=L - 1, groups=v.shape[1])[..., :L]
                + self.long_conv_bias.unsqueeze(-1) * z
            )
            z = z.to(v.dtype)
            z = q * z
        elif self.use_cgcg and self.use_medium_hyena:
            # TODO: if the conditions are met, we should not rearrange to l last in the first place
            dtype = q.dtype
            q = rearrange(q, "b (d g) l -> b l g d", g=self.num_groups)
            k = rearrange(k, "b (d g) l -> b l g d", g=self.num_groups)
            v = rearrange(v, "b (d g) l -> b l g d", g=self.num_groups)
            z = self.cgcg_fn.apply(
                q.to(self.cgcg_dtype),
                k.to(self.cgcg_dtype),
                v.to(self.cgcg_dtype),
                h[:, None].to(self.cgcg_dtype),  # g, 1, filter_l
                "default",
                False,
                self.cgcg_fwd_config,
                self.cgcg_bwd_config,
            )
            z = rearrange(z, "b l g d -> b (g d) l").to(dtype)
            # TODO: fix these hacky rearranges
            return rearrange(z, "b (np hn) sq -> b np sq hn", np=np)
        else:
            h = h.repeat_interleave(self.group_dim, dim=-2)

            if self.global_config.use_flashfft:
                # squeeze h dim (kernel), to get rid of leading 1 dim
                h = h.squeeze(0)
                z = self.fftconv_fn(v, h, k, q)
            else:
                z = k * v
                with torch.autocast("cuda"):
                    z = fftconv_func(
                        z.to(torch.float32),
                        h.to(torch.float32),
                        self.long_conv_bias,
                        None,
                        gelu=False,
                        bidirectional=self.bidirectional,
                    )
                    z = z.to(v.dtype)
                z = q * z

        if downsampled:
            z = z.repeat_interleave(self.downsample_factor, dim=-1)

        return rearrange(z, "b (np hn) sq -> b np sq hn", np=np)


class ParallelShortHyenaOperator(nn.Module):
    def __init__(
        self,
        hidden_size,
        global_config,
        init_method,
        short_conv_class,
        use_fast_causal_conv=False,
        is_mlp=False,
    ):
        super().__init__()
        self.global_config = global_config
        self.is_mlp = is_mlp
        self.hidden_size = hidden_size
        self.cgcg_dtype = torch.float32
        self.use_cgcg_mlp = global_config.use_cgcg_mlp
        self.use_cgcg_short = global_config.use_cgcg_short

        world_size = mpu.get_model_parallel_world_size()

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
            repeat_h_dg=not self.use_cgcg_mlp and not self.use_cgcg_short,
        )

        if is_mlp:
            if self.use_cgcg_mlp:
                self.cgcg_fn = TwoPassChunkedGateConvGate

            self.cgcg_fwd_config = FwdKernelConfig(
                schedule="default",
                CHUNK_SIZE=128,
                BLOCK_D=min(self.group_dim, 128),
                THREADBLOCK_SWIZZLE="row",
                name="fwd",
                version="v2",
                RETURN_TOEPLITZ=True,
                RETURN_Y2=True,
                RETURN_BX_LAG=True,
                num_warps=8,
            )

            self.cgcg_bwd_config = BwdKernelConfig(
                schedule="default",
                CHUNK_SIZE=128,
                BLOCK_D=min(self.group_dim, 128),
                THREADBLOCK_SWIZZLE="row",
                name="bwd",
                version="v2",
                LOAD_TOEPLITZ=True,
                LOAD_BX_LAG=True,
                num_warps=8,
            )
        else:
            if self.use_cgcg_short:
                self.cgcg_fn = TwoPassChunkedGateConvGate

                self.cgcg_fwd_config = FwdKernelConfig(
                    schedule="default",
                    CHUNK_SIZE=128,
                    BLOCK_D=min(self.group_dim, 128),
                    THREADBLOCK_SWIZZLE="row",
                    name="fwd",
                    version="v2",
                    RETURN_TOEPLITZ=True,
                    RETURN_Y2=True,
                    RETURN_BX_LAG=True,
                    num_warps=8,
                )

                self.cgcg_bwd_config = BwdKernelConfig(
                    schedule="default",
                    CHUNK_SIZE=128,
                    BLOCK_D=min(self.group_dim, 128),
                    THREADBLOCK_SWIZZLE="row",
                    name="bwd",
                    version="v2",
                    LOAD_TOEPLITZ=True,
                    LOAD_BX_LAG=True,
                    num_warps=8,
                )

    def forward(self, query_layer, key_layer, value_layer):
        # input sizes: [sq, b, np, hn]
        # seqlen, batch, tensor parallel, hidden size per tensor parallel
        np = query_layer.shape[-2]
        L = query_layer.shape[0]

        query = rearrange(query_layer, "sq b np hn -> b (np hn) sq")
        key = rearrange(key_layer, "sq b np hn -> b (np hn) sq")
        value = rearrange(value_layer, "sq b np hn -> b (np hn) sq")

        q, k, v = query[..., :L], key[..., :L], value[..., :L]

        if self.use_cgcg_mlp or self.use_cgcg_short:
            # TODO: if the conditions are met, we should not rearrange to l last in the first place
            dtype = q.dtype
            q = rearrange(q, "b (d g) l -> b l g d", g=self.num_groups)
            k = rearrange(k, "b (d g) l -> b l g d", g=self.num_groups)
            v = rearrange(v, "b (d g) l -> b l g d", g=self.num_groups)

            z = self.cgcg_fn.apply(
                q.to(self.cgcg_dtype),
                k.to(self.cgcg_dtype),
                v.to(self.cgcg_dtype),
                self.short_conv.short_conv_weight[:, None].to(self.cgcg_dtype),  # g, 1, filter_l
                "default",
                False,
                self.cgcg_fwd_config,
                self.cgcg_bwd_config,
            )
            z = rearrange(z, "b l g d -> b (g d) l").to(dtype)
            # TODO: fix these hacky rearranges
            return rearrange(z, "b (np hn) sq -> b np sq hn", np=np)

        else:
            if self.pregate:
                z = k * v
            else:
                z = v

            z = self.short_conv(z)

            if self.postgate:
                z = q * z
            else:
                z = z

        return rearrange(z, "b (np hn) sq -> b np sq hn", np=np)
