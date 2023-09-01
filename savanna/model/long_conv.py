import math
import os
import sys

from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from torch.nn.parameter import Parameter

from einops import rearrange, repeat

from savanna import mpu


@torch.jit.script
def _mul_sum(y, q):
    return (y * q).sum(dim=1)


def fftconv_heads_ref(k, ssm_kernel, D, q, v, head_dim=1, ssm_kernel_rev=None):
    seqlen = k.shape[-1]
    fft_size = 2 * seqlen
    kv = rearrange(k, "b (h d1) l -> b d1 1 h l", d1=head_dim) * rearrange(
        v, "b (h d2) l -> b 1 d2 h l", d2=head_dim
    )  # b d1 d2 h l
    kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
    ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
    if ssm_kernel_rev is not None:
        ssm_kernel_rev_f = torch.fft.rfft(ssm_kernel_rev, n=fft_size)  # h L+1
        ssm_kernel_f = ssm_kernel_f + ssm_kernel_rev_f.conj()
    y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm="forward")[
        ..., :seqlen
    ]  # b d1 d2 h l
    out = y + kv * D.unsqueeze(-1)  # b d1 d2 h l
    q = rearrange(q, "b (h d1) l -> b d1 1 h l", d1=head_dim)
    if head_dim > 1:
        out = _mul_sum(out, q)
        return rearrange(out, "b d2 h l -> b (h d2) l").to(dtype=k.dtype)
    else:
        return rearrange(out * q, "b 1 1 h l -> b h l").to(dtype=k.dtype)


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with mpu.get_cuda_rng_tracker().fork():
        init_method(weight)


# def mul_sum(q, y):
#     return (q * y).sum(dim=1)


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


def fftconv_heads_ref(k, ssm_kernel, D, q, v, head_dim=1, ssm_kernel_rev=None):
    seqlen = k.shape[-1]
    fft_size = 2 * seqlen
    kv = rearrange(k, "b (h d1) l -> b d1 1 h l", d1=head_dim) * rearrange(
        v, "b (h d2) l -> b 1 d2 h l", d2=head_dim
    )  # b d1 d2 h l
    kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
    ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
    if ssm_kernel_rev is not None:
        ssm_kernel_rev_f = torch.fft.rfft(ssm_kernel_rev, n=fft_size)  # h L+1
        ssm_kernel_f = ssm_kernel_f + ssm_kernel_rev_f.conj()
    y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm="forward")[
        ..., :seqlen
    ]  # b d1 d2 h l
    out = y + kv * D.unsqueeze(-1)  # b d1 d2 h l
    q = rearrange(q, "b (h d1) l -> b d1 1 h l", d1=head_dim)
    if head_dim > 1:
        out = _mul_sum(out, q)
        return rearrange(out, "b d2 h l -> b (h d2) l").to(dtype=k.dtype)
    else:
        return rearrange(out * q, "b 1 1 h l -> b h l").to(dtype=k.dtype)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        self.register_buffer(
            "t", torch.linspace(0, 1, self.seq_len)[None, :, None]
        )  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        # self.z = nn.Parameter(torch.cat([self.t, z.real, z.imag], dim=-1))
        # fix to non-learnable
        z = torch.cat([self.t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z)
        # self.t = t
        # self.t = self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ParallelExponentialModulation(nn.Module):
    def __init__(
        self,
        global_config,
        d_model,
        hidden_size_per_partition,
        mp_rank,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct

        self.weight = Parameter(
            torch.empty(
                1, 1, hidden_size_per_partition, dtype=global_config.params_dtype
            )
        )

        self.weight.model_parallel = True
        self.weight.partition_dim = 2
        self.weight.partition_stride = 1

        master_weight = torch.linspace(min_decay, max_decay, d_model)[None, None].to(
            global_config.params_dtype
        )

        weight_list = torch.split(master_weight, hidden_size_per_partition, dim=-1)
        rank = mpu.get_model_parallel_rank()
        world_size = mpu.get_model_parallel_world_size()
        my_weight_list = weight_list[rank::world_size]

        with torch.no_grad():
            torch.cat(my_weight_list, dim=self.weight.partition_dim, out=self.weight)
            # print(self.weight.shape)
            # print('hidden size per partition ', hidden_size_per_partition)

    def forward(self, t, x):
        decay = torch.exp(-t * self.weight.abs())
        # print('x ', x)
        # print('x shape ', x.shape)
        # print('decay shape ', decay.shape)
        x = x * (decay + self.shift)
        return x


class ParallelHyenaFilter(nn.Module):
    def __init__(
        self,
        global_config,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        num_inner_mlps=1,
        modulate: bool = True,
        normalized=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(d_model, world_size)

        # print('d model, world size, hidden size ', d_model, world_size, self.hidden_size_per_partition)

        self.act = Sin(dim=order, w=w)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)

        # uses a variable number of inner linear layers

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            self.act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(self.act)

        # final linear layer
        self.final_filter = mpu.ColumnParallelLinear(
            global_config=global_config,
            input_size=order,
            output_size=d_model,
            gather_output=False,
            init_method=init.xavier_normal_,
            bias=False,
        )

        self.modulation = ParallelExponentialModulation(
            global_config,
            d_model,
            self.hidden_size_per_partition,
            mpu.get_model_parallel_rank(),
            **kwargs,
        )

        self.normalized = normalized

    def forward(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        # print('h shape before final filter ', h.shape)
        h, _ = self.final_filter(h)
        # print('t ', t)
        # print('h shape after final filter ', h.shape)
        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        h = rearrange(h, "1 L D -> D (1 L)")
        return h


class ParallelHyenaConv(nn.Module):
    """
    Inner action for a parallel Hyena Conv (replacing attention).
    Testing some new improvements...

    Inputs: Q, K, V
    Operation: 1D Conv on each Q, K, V (independently)
    Long Conv(Q * K) * V, independently on each H
    """

    def __init__(
        self,
        global_config,
    ):
        super().__init__()

        self.fp16 = global_config.precision == "fp16"
        self.bf16 = global_config.precision == "bfloat16"
        self.act = global_config.precision.gating_act

        self.use_hyena_filter = global_config.use_hyena_filter

        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(
            global_config.hidden_size, world_size
        )

        self.L = global_config.seq_length
        self.num_heads = global_config.num_heads
        self.head_dim = self.hidden_size_per_partition // self.num_heads
        self.short_conv_L = global_config.short_conv_L

        init_method = init.xavier_normal_

        self.short_conv_weight = nn.Parameter(
            torch.empty(
                global_config.precision.short_conv_len,  # Q K V
                self.hidden_size_per_partition,
                1,
                self.short_conv_L,
                device=torch.cuda.current_device(),
                dtype=global_config.params_dtype,
            )
        )

        self.short_conv_bias = nn.Parameter(
            torch.empty(
                3,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=global_config.params_dtype,
            )
        )
        self.short_conv_bias.model_parallel = True
        self.short_conv_bias.partition_dim = 1
        self.short_conv_bias.stride = 1

        _initialize_affine_weight_gpu(
            self.short_conv_weight, init_method, partition_dim=1
        )

        if not self.use_hyena_filter:
            self.long_conv_weight = nn.Parameter(
                torch.empty(
                    self.hidden_size_per_partition,
                    self.L,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )

            _initialize_affine_weight_gpu(
                self.long_conv_weight, init_method, partition_dim=0
            )
        else:
            self.filter = ParallelHyenaFilter(
                global_config,
                d_model=global_config.hidden_size // self.num_heads,
                emb_dim=global_config.hyena_filter_emb_dim,
                order=global_config.hyena_filter_order,
                num_inner_mlps=global_config.hyena_filter_num_inner_mlps,
                seq_len=self.L,
                w=global_config.hyena_filter_w,
            )

        self.long_conv_bias = nn.Parameter(
            torch.empty(
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            )
        )

        self.long_conv_bias.model_parallel = True
        self.long_conv_bias.partition_dim = 0
        self.long_conv_bias.stride = 1

    def forward(self, query_layer, key_layer, value_layer):
        # input sizes: [sq, b, np, hn]
        # seqlen, batch, tensor parallel, hidden size per tensor parallel
        np = query_layer.shape[-2]

        query = rearrange(query_layer, "sq b np hn -> b (np hn) sq")
        key = rearrange(key_layer, "sq b np hn -> b (np hn) sq")
        value = rearrange(value_layer, "sq b np hn -> b (np hn) sq")

        q = F.conv1d(
            query,
            self.short_conv_weight[0, :, :, :],
            self.short_conv_bias[0, :],
            stride=1,
            padding=self.short_conv_L - 1,
            dilation=1,
            groups=self.short_conv_weight.shape[1],
        )[..., : self.L]
        k = F.conv1d(
            key,
            self.short_conv_weight[1, :, :, :],
            self.short_conv_bias[1, :],
            stride=1,
            padding=self.short_conv_L - 1,
            dilation=1,
            groups=self.short_conv_weight.shape[1],
        )[..., : self.L]
        v = F.conv1d(
            value,
            self.short_conv_weight[2, :, :, :],
            self.short_conv_bias[2, :],
            stride=1,
            padding=self.short_conv_L - 1,
            dilation=1,
            groups=self.short_conv_weight.shape[1],
        )[..., : self.L]

        filter = self.filter(self.L)
        filter = filter.repeat_interleave(self.num_heads, dim=0)
        z = v * self.act(k)
        with torch.autocast("cuda"):
            z = fftconv_ref(
                z.to(torch.float32),
                filter.to(torch.float32),
                self.long_conv_bias,
                None,
                gelu=False,
            )
            z = z.to(v.dtype)
        z = z * self.act(q)
        return rearrange(z, "b (np hn) sq -> b np sq hn", np=np)


class ParallelBadgerConv(nn.Module):
    """
    Inner action for a parallel Badger Conv (replacing attention).

    Inputs: Q, K, V
    Operation: 1D Conv on each Q, K, V (independently)
    Long Conv(Q * K) * V, independently on each H
    """

    def __init__(
        self,
        global_config,
    ):
        super().__init__()

        self.fp16 = global_config.precision == "fp16"
        self.bf16 = global_config.precision == "bfloat16"

        self.use_hyena_filter = global_config.use_hyena_filter

        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()
        # print(self.model_parallel_rank)

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(
            global_config.hidden_size, world_size
        )

        self.L = global_config.seq_length
        self.num_heads = global_config.num_heads
        self.head_dim = self.hidden_size_per_partition // self.num_heads
        self.short_conv_L = global_config.short_conv_L

        init_method = init.xavier_normal_

        self.short_conv_weight = nn.Parameter(
            torch.empty(
                3,  # Q K V
                self.hidden_size_per_partition,
                1,
                self.short_conv_L,
                device=torch.cuda.current_device(),
                dtype=global_config.params_dtype,
            )
        )

        # print(self.model_parallel_rank, self.short_conv_weight.shape)
        self.short_conv_bias = nn.Parameter(
            torch.empty(
                3,
                self.hidden_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=global_config.params_dtype,
            )
        )
        self.short_conv_bias.model_parallel = True
        self.short_conv_bias.partition_dim = 1
        self.short_conv_bias.stride = 1

        _initialize_affine_weight_gpu(
            self.short_conv_weight, init_method, partition_dim=1
        )

        if not self.use_hyena_filter:
            self.long_conv_weight = nn.Parameter(
                torch.empty(
                    self.hidden_size_per_partition,
                    self.L,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )

            _initialize_affine_weight_gpu(
                self.long_conv_weight, init_method, partition_dim=0
            )
        else:
            self.filter = ParallelHyenaFilter(
                global_config,
                d_model=global_config.hidden_size // self.num_heads,
                emb_dim=global_config.hyena_filter_emb_dim,
                order=global_config.hyena_filter_order,
                num_inner_mlps=global_config.hyena_filter_num_inner_mlps,
                seq_len=self.L,
                w=global_config.hyena_filter_w,
            )

        self.long_conv_bias = nn.Parameter(
            torch.empty(
                self.head_dim,
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            )
        )

        self.long_conv_bias.model_parallel = True
        self.long_conv_bias.partition_dim = 0
        self.long_conv_bias.stride = 1

    def forward(self, query_layer, key_layer, value_layer):
        # input sizes: [sq, b, np, hn]
        # seqlen, batch, tensor parallel, hidden size per tensor parallel
        np = query_layer.shape[-2]

        # print('query layer: ', query_layer.shape, self.model_parallel_rank, query_layer.device)

        query = rearrange(query_layer, "sq b np hn -> b (np hn) sq")
        key = rearrange(key_layer, "sq b np hn -> b (np hn) sq")
        value = rearrange(value_layer, "sq b np hn -> b (np hn) sq")

        # print('query: ', query.shape, self.model_parallel_rank, query.device)
        # print('short conv weight: ', self.short_conv_weight.shape, self.model_parallel_rank, self.short_conv_weight.device)

        # run the conv1D
        query = F.conv1d(
            query,
            self.short_conv_weight[0, :, :, :],
            self.short_conv_bias[0, :],
            stride=1,
            padding=self.short_conv_L - 1,
            dilation=1,
            groups=self.short_conv_weight.shape[1],
        )[..., : self.L]
        key = F.conv1d(
            key,
            self.short_conv_weight[1, :, :, :],
            self.short_conv_bias[1, :],
            stride=1,
            padding=self.short_conv_L - 1,
            dilation=1,
            groups=self.short_conv_weight.shape[1],
        )[..., : self.L]
        value = F.conv1d(
            value,
            self.short_conv_weight[2, :, :, :],
            self.short_conv_bias[2, :],
            stride=1,
            padding=self.short_conv_L - 1,
            dilation=1,
            groups=self.short_conv_weight.shape[1],
        )[..., : self.L]

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        filter = self.filter(self.L)

        with torch.autocast("cuda"):
            y = fftconv_heads_ref(
                value.to(torch.float32),
                filter.to(torch.float32),
                self.long_conv_bias.to(torch.float32),
                v=key,
                head_dim=self.num_heads,
                q=query,
            )
            y = y.to(value.dtype)

        return rearrange(y, "b (np hn) sq -> b np sq hn", np=np)
