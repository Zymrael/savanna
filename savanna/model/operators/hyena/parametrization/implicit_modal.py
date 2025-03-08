# Copyright (c) 2024, Michael Poli, Eric Nguyen

import math

import torch
import torch.nn as nn
from einops import rearrange

from savanna import print_rank_0


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
        dt = torch.rand(self.order, self.d_model) * (dt_max - dt_min) + dt_min
        self.register_no_wd_param("dt", nn.Parameter(dt))
        self.init_ranges = torch.randn(order, pole_factors)

        self.register_no_wd_param("p", nn.Parameter(self.init_ranges[:,None,:].repeat(1, d_model, 1)))
        self.register_no_wd_param("R", nn.Parameter(torch.randn(order, d_model, residue_factors) / math.sqrt(d_model)))
        self.t = torch.linspace(0, L_cache, L_cache)[:1].unsqueeze(0).unsqueeze(0).float()
        self.residues_norm = nn.LayerNorm(d_model)

        # self.register_no_wd_param("h_0", nn.Parameter(torch.randn(d_model, 1)))
        self.initialize_poles_and_residues()

    def initialize_poles_and_residues(self):
        self.R.data = torch.randn(self.order, self.d_model, self.residue_factors) / math.sqrt(self.order)

    def get_clamped_dt(self, L):
            return torch.clamp(self.dt, min=self.dt_min, max=self.dt_max)[..., None]

    def forward(self, L, *args, **kwargs):
        return self.filter(L, *args, **kwargs)

    def get_t(self, L):
        if L < self.t.size(0) + 1:
            t = self.t[..., :L]
        else:
            t = torch.linspace(0, L, L ).unsqueeze(0).unsqueeze(0).to(self.p)
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



class ParallelComplexModalFilter(nn.Module):
    def __init__(
        self,
        global_config,
        d_model: int,
        order: int,
        mimo: bool = False,
        theta_max: float = 2 * torch.pi,
    ):
        """
        Stability of the system in encouraged by init but not enforced.
        Args:
            d_model: dimension of the input and output (number of channels)
            order: order of the filter (number of states of underlying state-space model)
            mimo: whether the filter is MIMO or SISO
            r_min: minimum radius of the poles (init hyperparameter)
            r_max: maximum radius of the poles (init hyperparameter)
            theta_max: maximum phase of the poles (init hyperparameter)
        """
        super().__init__()
        self.order = order
        self.d_model = d_model
        self.mimo = mimo

        # Init poles and residues
        self.register_parameter("r", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("theta", nn.Parameter(torch.ones(order // 2, d_model)))

        if mimo:
            # TODO: implement MIMO case where R is a tensor of shape (order // 2, d_model, d_model)
            raise NotImplementedError
        self.register_parameter("R_re", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("R_im", nn.Parameter(torch.ones(order // 2, d_model)))

        self.register_parameter("h_0", nn.Parameter(torch.ones(1, d_model)))
        r_min, r_max = (
            global_config.hyena_filter_r_min,
            global_config.hyena_filter_r_max,
        )
        self._init_params(r_max, r_min, theta_max)

    def _init_params(self, r_max, r_min, theta_max):
        # Init poles distributed uniformly of ring of the complex plane
        # between r_min and r_max and phase between 0 and theta_max
        u1 = torch.rand(self.order // 2, self.d_model)
        u2 = torch.rand(self.order // 2, self.d_model)
        self.r.data = r_min + (r_max - r_min) * u1
        self.theta.data = theta_max * u2
        # Init residues with Glorot initialization
        self.R_re.data = torch.randn(self.order // 2, self.d_model) * math.sqrt(2 / self.order)
        self.R_im.data = torch.randn(self.order // 2, self.d_model) * math.sqrt(2 / self.order)

    def _get_poles_and_residues(self):
        # poles
        p = self.r * torch.exp(1j * self.theta)
        # residues
        R = self.R_re + 1j * self.R_im
        return p, R

    def compute_filter(self, L):
        p, R = self._get_poles_and_residues()
        t = torch.arange(L - 1).unsqueeze(1).unsqueeze(2).to(p)
        h = torch.sum(R * p**t, dim=1).real
        return h

    def forward(self, L, *args, **kwargs):
        # evaluate filter for t = 1, ..., L
        h = self.compute_filter(L)
        # stack h_0 to the beginning of the filter
        h = torch.cat([self.h_0.to(h), h], dim=0)
        h = rearrange(h, "L D -> D L")
        return h



# class ImplicitModalFilter(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         order=64,
#         L_cache=None,
#         dt_min=0.01,
#         dt_max=0.1,
#         lr=None,
#     ):
#         super().__init__()
#         self.order = order
#         self.d_model = d_model
#         dtype, cdtype = torch.float32, torch.cfloat

#         self.t = torch.linspace(0, L_cache, L_cache)[:1].unsqueeze(0).unsqueeze(0).float()

#         log_dt = torch.rand(self.d_model, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(
#                 dt_min
#             )

#         # small phase init 
#         logp_re = (torch.rand(d_model, order // 2)).log()
#         p_im = torch.rand(d_model, order // 2)

#         R = torch.randn(d_model, order // 2, dtype=torch.cfloat)
        
#         self.log_dt = nn.Parameter(log_dt)
#         self.logp_real = nn.Parameter(logp_re)
#         self.p_im = nn.Parameter(p_im)
#         self.R = nn.Parameter(R)

#     def get_t(self, L):
#         if L < self.t.size(0) + 1:
#             t = self.t[..., :L]
#         else:
#             t = torch.linspace(0, L, L ).unsqueeze(0).unsqueeze(0).to(self.logp_real)
#         return t
    
#     def compute_filter(self, L, t):
#         log_dt = self.log_dt.to(torch.float32)
#         t = t.to(torch.float32)
#         p = torch.exp(-torch.exp(self.logp_real))
#         print(p) 
#         pdt = (p.unsqueeze(-1)) ** t
#         h = pdt * self.R.unsqueeze(-1)
#         h = torch.sum(h, dim=1)[None]
#         return h, None


#     def filter(self, L, *args, **kwargs):
#         t = self.get_t(L)
#         h = self.compute_filter(L, t)
#         return h

#     def forward(self, L, **kwargs):
#         return self.filter(L)


class ImplicitModalFilter(nn.Module):
    def __init__(
            self,
            d_model,
            order=64,
            L_cache=None,
            gamma_min=0.01,
            gamma_max=0.1,
            lr=None,
    ):
        super().__init__()
        self.order = order
        self.d_model = d_model
        t = rearrange(torch.arange(L_cache, dtype=torch.float32), "L -> 1 1 L")  # <- this should be arange
        self.register_buffer("t", t)
        self.use_cached_t = False

        gamma = torch.rand(self.d_model, order, dtype=torch.float32) * (gamma_max - gamma_min) + gamma_min
        gamma = gamma.log()
        self.gamma = nn.Parameter(gamma)

        R = 1e-1 * torch.randn(d_model, order, dtype=torch.float32) / math.sqrt(order)
        self.R = nn.Parameter(R)
        self.p = nn.Parameter(-torch.ones(d_model, order, dtype=torch.float32))

    def get_t(self, L):
        #print_rank_0(f"DEBUG::IMPLICIT_MODAL_FILTER::get_t: {L=} {self.t.shape=} {self.use_cached_t=}")

        if L < self.t.size(
                -1):  # <- this should check the last dimension, which is the actual "sequence length" dimension
            t = self.t[..., :L].to(torch.float32)
        else:
            t = rearrange(torch.arange(L, dtype=torch.float32, device=self.t.device), "L -> 1 1 L")
            self.t = t
        return t

        # Assumes L <= L_cache
        # if self.use_cached_t:
        #     return self.t[..., :L]

        # t = rearrange(torch.arange(L, dtype=torch.float32, device=self.t.device), "L -> 1 1 L")
        # self.t = t
        # self.use_cached_t = True

        # return t

        
    def compute_filter(self, L, t):
        assert t.dtype == torch.float32, f't must be float32. Current dtype: {t.dtype}'
        #print_rank_0(f"DEBUG::IMPLICIT_MODAL_FILTER::compute_filter: {L=} {self.p.shape=} {self.gamma.shape=} {self.R.shape=} {t.shape=}")
        logp = -torch.exp(self.p.to(torch.float32))
        glogp = logp * torch.exp(self.gamma.to(torch.float32))
        h = torch.exp(glogp[..., None] * t)
        #print(f"DEBUG::IMPLICIT_MODAL_FILTER::compute_filter: {h.shape=}")
        h = torch.einsum('do,dot->dt', self.R.to(torch.float32), h)
        h = h[None]
       # print(f"DEBUG::IMPLICIT_MODAL_FILTER::compute_filter:final {h.shape=}")
        return h, None

    def filter(self, L, *args, **kwargs):
        t = self.get_t(L)
        h = self.compute_filter(L, t)
        return h

    def forward(self, L, **kwargs):
        return self.filter(L)
