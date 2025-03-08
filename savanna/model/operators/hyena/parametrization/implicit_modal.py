# Copyright (c) 2024, Michael Poli, Eric Nguyen

import torch
import torch.nn as nn
import math
from savanna.model.operators.hyena.parametrization.utils import get_dtype_from_string
from savanna import mpu
from einops import rearrange, repeat


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
        We parametrize the poles and residues, i.e. the matrices A and C, of the state-space model. B is fixed to a one-vector and D is learned as a bias.
        The poles are parametrized as complex numbers in polar form, i.e. r * exp(i * theta), where r is the radius and theta the phase.
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
