# Copyright (c) 2024, Michael Poli, Eric Nguyen

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from opt_einsum import contract


class ExplicitDecayFilter(nn.Module):
    def __init__(self, r_min, r_max, d_model, order, L_cache):
        super().__init__()
        self.order = order
        self.d_model = d_model
        self.L_cache = L_cache
        # self.register_no_wd_param("r", nn.Parameter(torch.ones(order // 2, d_model)))
        # self.register_no_wd_param("R_re", nn.Parameter(torch.ones(order // 2, d_model)))
        # self.t = torch.linspace(0, 1, L_cache - 1).unsqueeze(0).unsqueeze(0).float()

        # self.register_no_wd_param("h_0", nn.Parameter(torch.ones(d_model, 1)))
        # r_min, r_max = (
        #     r_min,
        #     r_max,
        # )
        # self._init_params(r_max, r_min)

    def _init_params(self, r_max, r_min):
        u1 = torch.rand(self.order // 2, self.d_model)
        self.r.data = r_min + (r_max - r_min) * u1
        self.R_re.data = torch.randn(self.order // 2, self.d_model) / math.sqrt(self.order // 2)

    def compute_filter(self, L):
        p, R = self.r, self.R_re
        if L < self.t.size(0):
            t = self.t[..., :L]
        else:
            t = torch.arange(L - 1).unsqueeze(0).unsqueeze(0).to(p)
        log_p_exp = t * p.unsqueeze(-1).log()
        p_exp = torch.exp(log_p_exp)
        h = torch.sum(R.unsqueeze(-1) * p_exp, dim=0)
        return h

    def filter(self, L, *args, **kwargs):
        h = self.compute_filter(L)
        h = torch.cat([self.h_0.to(h), h], dim=1)
        return h

    def register_no_wd_param(self, name, tensor, lr=None):
        self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0.0}
        setattr(getattr(self, name), "_optim", optim)
