# Copyright (c) 2024, Michael Poli, Eric Nguyen

import torch 
import torch.nn as nn

from savanna import mpu
from savanna.model.operators.hyena.parametrization.implicit_utils import (
    PositionalEmbedding,
    RandomFourierPositionalEmbedding,
    ParallelExponentialModulation,
)
from savanna.model.activations import Sin


class ParallelImplicitFreeformFilter(nn.Module):
    def __init__(
        self,
        global_config,
        init_method,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        w=1,  # frequency of periodic activations
        omega_0=1,  # frequency of positional embeddings
        wd=0,  # weight decay of kernel parameters
        num_inner_mlps=2,
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
        self.bidirectional = global_config.bidirectional

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(d_model, world_size)

        self.act = Sin(dim=order, w=w)

        if global_config.hyena_pos_emb == "fourier_fixed":
            assert (
                emb_dim % 2 != 0 and emb_dim >= 3
            ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
            self.pos_emb = PositionalEmbedding(emb_dim, seq_len, dtype=global_config.precision)

        elif global_config.hyena_pos_emb == "random_fourier":
            self.pos_emb = RandomFourierPositionalEmbedding(emb_dim, seq_len, omega_0)

        if self.bidirectional:
            d_model = 2 * d_model

        # uses a variable number of inner linear layers
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            self.act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order, bias=True))
            self.implicit_filter.append(self.act)

        final_init_method = torch.nn.init.xavier_normal_

        # final linear layer

        self.final_filter = nn.Linear(order, d_model, bias=False)
        torch.nn.init.xavier_normal_(self.final_filter.weight, gain=1)
        # self.final_filter = mpu.ColumnParallelLinear(
        #     global_config=global_config,
        #     input_size=order,
        #     output_size = d_model,
        #     gather_output=False,
        #     init_method = init_method,
        #     bias = False
        # )
        fast_decay_pct, slow_decay_pct = (
            global_config.hyena_filter_fast_decay,
            global_config.hyena_filter_slow_decay,
        )
        self.modulation = ParallelExponentialModulation(
            global_config,
            d_model,
            self.hidden_size_per_partition,
            mpu.get_model_parallel_rank(),
            fast_decay_pct=fast_decay_pct,
            slow_decay_pct=slow_decay_pct,
            **kwargs,
        )

        self.normalized = normalized
        if normalized:
            self.post_modulation_norm = nn.LayerNorm(d_model)

    def forward(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.final_filter(h)

        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            # h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
            h = self.post_modulation_norm(h)

        h = rearrange(h, "1 L D -> D (1 L)")
        return h
