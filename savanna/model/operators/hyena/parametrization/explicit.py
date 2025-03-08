# Copyright (c) 2024, Michael Poli, Eric Nguyen

"""
Family of explicit positional (time-varying) parametrizations for convolutional filters in Hyena cascade models
"""
import math

import torch
import torch.nn as nn

from savanna import mpu, print_rank_0
from savanna.utils import ALLOC_DEVICE


class ExplicitSingleDecayFilter(nn.Module):
    def __init__(
        self,
        d_model,
        L_cache,
        log_r_min=0,
        log_r_max=2,
        unit_passthrough=False,
        decay_preset="strong",
        num_decay_repeats=1,
        small_init=True,
    ):
        super().__init__()
        print_rank_0(f"Initializing ExplicitSingleDecayFilter with d_model={d_model}, L_cache={L_cache}")
        assert decay_preset in ["strong", "normal", "weak"]
        if decay_preset == "strong":
            log_r_min = 0
            log_r_max = 2
        elif decay_preset == "normal":
            log_r_min = -1
            log_r_max = 2
        elif decay_preset == "weak":
            log_r_min = -2
            log_r_max = 2

        h = torch.randn(d_model, L_cache, device=ALLOC_DEVICE) / math.sqrt(L_cache)
        if small_init:
            h = h * 1e-5
        if unit_passthrough:
            h[:, :1] = 1.0
        self.h = nn.Parameter(h)
        t = torch.linspace(0, 1, L_cache)[None]

        self.log_r_min = log_r_min
        self.log_r_max = log_r_max
        self.num_decay_repeats = num_decay_repeats
        self.model_parallel_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = mpu.get_model_parallel_rank()

        # @mp: d_model passed to ExplicitSingleDecayFilter is the dimension per model parallel rank
        # to ensure consistency across different model parallel world sizes, we initialize decays
        # for the global model dimension, then slice appropriately for each model parallel rank
        global_d_model = d_model * self.model_parallel_size // self.num_decay_repeats
        decay_domain = torch.logspace(log_r_min, log_r_max, global_d_model)[:, None].repeat(
            self.num_decay_repeats, 1
        )
        decay_domain = decay_domain[
            self.model_parallel_rank * d_model : (self.model_parallel_rank + 1) * d_model, :
        ]
        decay = torch.exp(-decay_domain * t)

        self.register_buffer("decay", decay)

    def forward(self, L, *args, **kwargs):
        return self.filter(L, *args, **kwargs)

    @torch.compile(mode="max-autotune")
    def filter(self, L, *args, **kwargs):
        h = self.h[:, :L]
        h = h * self.decay[:, :L]
        return h
