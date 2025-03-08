# Copyright (c) 2024, Michael Poli, Eric Nguyen

"""
Family of explicit positional (time-varying) parametrizations for convolutional filters in Hyena cascade models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from savanna.utils import ALLOC_DEVICE

from savanna import print_rank_0

class ExplicitSingleDecayFilter(nn.Module):
    def __init__(self,
            d_model,
            L_cache,
            log_r_min=0,
            log_r_max=2,
            unit_passthrough=False,
            decay_preset="strong",
            small_init=True):
        super().__init__()
        print_rank_0(f"Initializing ExplicitSingleDecayFilter with d_model={d_model}, L_cache={L_cache}")
        h = torch.randn(d_model, L_cache, device=ALLOC_DEVICE) / math.sqrt(L_cache)
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

        if small_init:
            h = h * 1e-5
        if unit_passthrough:
            h[:, :1] = 1.0
        self.h = nn.Parameter(h)
        t = torch.linspace(0, 1, L_cache)[None]
        self.log_r_min = log_r_min
        self.log_r_max = log_r_max
        decay = torch.logspace(log_r_min, log_r_max, d_model)[:, None]
        decay = torch.exp(- decay * t)
        self.register_buffer("decay", decay)

    def forward(self, L, *args, **kwargs):
        return self.filter(L, *args, **kwargs)

    @torch.compile(mode="max-autotune")
    def filter(self, L, *args, **kwargs):
        h = self.h[:, :L]
        h = h * self.decay[:, :L]
        return h


class ImplicitRealModalFilter(nn.Module):
    def __init__(self,
            d_model,
            order,
            L_cache,
            dt_min=0.001,
            dt_max=0.1,
            p_min=0.01,
            p_max=0.99,
            residue_factors=4,
            pole_factors=4):
        super().__init__()
        self.order = order
        self.d_model = d_model
        self.L_cache = L_cache
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.residue_factors = residue_factors

        # initialize as follows: bucket init range into d_model buckets,
        # then split each bucket into `order` segments
        #self.init_ranges = torch.linspace(-8, 5, d_model * order).reshape(order, d_model)
        # self.init_ranges = torch.logspace(-2, 0, d_model * order).reshape(order, d_model)
        # self.init_ranges = torch.logspace(-1, 0, d_model * order).reshape(order, d_model)
        dt = torch.rand(self.order, self.d_model, device=ALLOC_DEVICE) * (dt_max - dt_min) + dt_min
        self.register_no_wd_param("dt", nn.Parameter(dt))
        self.init_ranges = torch.randn(order, pole_factors, device=ALLOC_DEVICE)

        self.register_no_wd_param("p", nn.Parameter(self.init_ranges[:,None,:].repeat(1, d_model, 1)))
        self.register_no_wd_param("R", nn.Parameter(torch.randn(order, d_model, residue_factors) / math.sqrt(d_model)))
        self.t = torch.linspace(0, L_cache, L_cache, device=ALLOC_DEVICE)[:1].unsqueeze(0).unsqueeze(0).float()
        self.residues_norm = nn.LayerNorm(d_model)

        # self.register_no_wd_param("h_0", nn.Parameter(torch.randn(d_model, 1)))
        self.initialize_poles_and_residues()

    def initialize_poles_and_residues(self):
        self.R.data = torch.randn(self.order, self.d_model, self.residue_factors, device=ALLOC_DEVICE) / math.sqrt(self.order)

    def get_clamped_dt(self, L):
            return torch.clamp(self.dt, min=self.dt_min, max=self.dt_max)[..., None]

    def forward(self, L, *args, **kwargs):
        return self.filter(L, *args, **kwargs)

    def get_t(self, L):
        if L < self.t.size(0) + 1:
            t = self.t[..., :L]
        else:
            t = torch.linspace(0, L, L, device=self.p.device).unsqueeze(0).unsqueeze(0).to(dtype=self.p.dtype)
        return t

    @torch.compile(mode="max-autotune")
    def compute_filter(self, L, t):
        R = self.R.prod(dim=-1)

        dt = self.get_clamped_dt(L)
        p = torch.exp(-torch.exp(self.p.prod(dim=-1)))
        log_ptdt = t * dt * p.unsqueeze(-1).log()
        pt = torch.exp(log_ptdt)
        h = torch.sum(R.unsqueeze(-1) * pt, dim=0)
        return h

    def filter(self, L, *args, **kwargs):
        t = self.get_t(L)
        h = self.compute_filter(L, t)
        return h

    def register_no_wd_param(self, name, tensor, lr=None):
        self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0.0}
        setattr(getattr(self, name), "_optim", optim)


# class ImplicitRealModalFilter(nn.Module):
#     def __init__(self,
#             d_model,
#             order,
#             L_cache,
#             residue_factors=4):
#         super().__init__()
#         self.order = order
#         self.d_model = d_model
#         self.L_cache = L_cache
#         self.residue_factors = residue_factors

#         # initialize as follows: bucket init range into d_model buckets,
#         # then split each bucket into `order` segments
#         #self.init_ranges = torch.linspace(-8, 5, d_model * order).reshape(order, d_model)
#         # self.init_ranges = torch.logspace(-2, 0, d_model * order).reshape(order, d_model)
#         self.init_ranges = torch.logspace(-1, 0, d_model * order).reshape(order, d_model)


#         dt = torch.rand(self.order, self.d_model) * (0.1 - 0.001) + 0.01
#         self.register_no_wd_param("dt", nn.Parameter(dt))
#         self.register_no_wd_param("p", nn.Parameter(self.init_ranges))
#         self.register_no_wd_param("R", nn.Parameter(torch.randn(order, d_model, residue_factors) / math.sqrt(d_model)))
#         self.t = torch.linspace(0, L_cache, L_cache)[:1].unsqueeze(0).unsqueeze(0).float()
#         self.residues_norm = nn.LayerNorm(d_model)

#         self.register_no_wd_param("h_0", nn.Parameter(torch.randn(d_model, 1)))
#         self.initialize_poles_and_residues()

#     def initialize_poles_and_residues(self):
#         u1 = torch.rand(self.order, self.d_model)
#         self.R.data = torch.randn(self.order, self.d_model, self.residue_factors) / math.sqrt(self.order)

#     def get_clamped_dt(self, L):
#             return torch.clamp(self.dt, min=0.001, max=0.1)[..., None]

#     def forward(self, L, *args, **kwargs):
#         return self.filter(L, *args, **kwargs)

#     def compute_filter(self, L):
#         if L < self.t.size(0) + 1:
#             t = self.t[..., :L]
#         else:
#             t = torch.linspace(0, L, L )[1:].unsqueeze(0).unsqueeze(0).to(self.p)
#         R = self.R.cumprod(dim=-1)
#         # R = self.residues_norm(R) / math.sqrt(self.order)
#         p = torch.clamp(self.p, min=0.01, max=0.99)

#         dt = self.get_clamped_dt(L)
#         log_p_exp = t * dt * p.unsqueeze(-1).log()
#         p_exp = torch.exp(log_p_exp)
#         h = torch.sum(R.unsqueeze(-1) * p_exp, dim=0)
#         return h

#     def filter(self, L, *args, **kwargs):
#         h = self.compute_filter(L)
#         h = torch.cat([self.h_0.to(h), h], dim=1)
#         return h

#     def register_no_wd_param(self, name, tensor, lr=None):
#         self.register_parameter(name, nn.Parameter(tensor))
#         optim = {"weight_decay": 0.0}
#         setattr(getattr(self, name), "_optim", optim)


class CascadeFilterParametrization(nn.Module):
    def __init__(
        self,
        filter_cls,
        d_model,
        L_cache,
        log_r_min=-2,
        log_r_max=2,
        small_init=True):
        super().__init__()
        pass
