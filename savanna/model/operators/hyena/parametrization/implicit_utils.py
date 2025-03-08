# Copyright (c) 2024, Michael Poli, Eric Nguyen

import torch
import math
import torch.nn as nn
from torch.nn import Parameter
from savanna import mpu

from savanna.dtype import get_dtype_from_string


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, dtype: str, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()
        self.dtype = get_dtype_from_string(dtype)
        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        self.register_buffer("t", torch.linspace(0, 1, self.seq_len)[None, :, None])  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        # self.z = nn.Parameter(torch.cat([self.t, z.real, z.imag], dim=-1))
        # fix to non-learnable
        z = torch.cat([self.t, z.real, z.imag], dim=-1).to(self.dtype)

        self.register_buffer("z", z)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class RandomFourierPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        seq_len: int,
        omega_0: float,
        use_bias: bool = False,
        **kwargs,
    ):
        if emb_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even. Current {emb_dim}")
        super().__init__()

        linear_out_channels = emb_dim // 2
        self.linear = torch.nn.Linear(in_features=1, out_features=linear_out_channels, bias=use_bias)
        # initialize with xavier normal rescaled by 0.02
        torch.nn.init.xavier_normal_(self.linear.weight, gain=0.02)

        # Initialize:
        self.linear.weight.data.normal_(0.0, 2 * torch.pi * omega_0)
        if use_bias:
            torch.nn.init.constant_(self.linear.bias, 0.0)

        t = torch.linspace(-1, 1, seq_len)[None, :, None]
        self.register_buffer("t", t)

    def forward(self, L):
        out = self.linear(self.t[:, :L])
        return torch.cat([torch.cos(out), torch.sin(out)], dim=-1), (self.t + 1) / 2


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
            torch.empty(1, 1, hidden_size_per_partition, dtype=global_config.params_dtype)
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

    def forward(self, t, x):
        decay = torch.exp(-t * self.weight.abs())
        x = x * (decay + self.shift)
        return x
