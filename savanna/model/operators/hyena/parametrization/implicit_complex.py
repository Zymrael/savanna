
# Copyright (c) 2024, Michael Poli, Eric Nguyen

import math
import torch
import torch.nn as nn

from einops import repeat



class ImplicitComplexModalFilter(nn.Module):

    def __init__(
        self,
        A,
        B,
        C,
        log_dt,
        L=None,
        lr=None,
    ):
        super().__init__()
        self.L = L
        self.d_model = log_dt.size(-1)
        self.order = A.size(-1)

        self.C = nn.Parameter(torch.view_as_real(C.conj().resolve_conj()))
        self.log_dt = nn.Parameter(log_dt)
        self.inv_A_real = nn.Parameter(torch.log(-A.real))
        self.A_imag = nn.Parameter(A.imag)
        self.B = nn.Parameter(torch.view_as_real(B.conj().resolve_conj()))


class ParallelComplexModalFilter(nn.Module):
    def __init__(
        self,
        d_model,
        order=64,
        L=None,
        dt_min=0.001,
        dt_max=0.1,
        lr=None,
    ):
        super().__init__()
        self.order = order
        self.d_model = d_model
        dtype, cdtype = torch.float, torch.cfloat

        log_dt = torch.rand(self.d_model, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(
                dt_min
            )

        assert dtype == torch.float or dtype == torch.double
        dtype = torch.cfloat if dtype == torch.float else torch.cdouble

        pi = torch.tensor(math.pi)
        real_part = 0.5 * torch.ones(d_model, order // 2)
        imag_part = repeat(torch.arange(order // 2), "order -> d_model order", d_model=d_model)

        imag_part = pi * imag_part
        
        A = -real_part + 1j * imag_part
        B = torch.ones(d_model, order // 2, dtype=dtype)
        C = torch.randn(1, self.d_model, self.order // 2, dtype=cdtype)    
        
        # The inner class is for backwards compatibility with legacy versions of the codebase
        self.inner_cls = ImplicitComplexModalFilter(
            A,
            B,
            C,
            log_dt,
            L=L,
            lr=lr,
        )

    def forward(self, L, state=None, u=None, **kwargs):
        
        log_dt = self.inner_cls.log_dt.to(torch.float32)
        dt = torch.exp(log_dt)

        C = torch.view_as_complex(self.inner_cls.C.to(torch.float32))
        B = torch.view_as_complex(self.inner_cls.B.to(torch.float32))  

        A_real = -torch.exp(self.inner_cls.inv_A_real)
        A_imag = self.inner_cls.A_imag
        A = A_real + 1j * self.inner_cls.A_imag
        dtA = A * dt.unsqueeze(-1)  

        #BC = (B[None, :, :] * C).view(-1, self.d_model, self.order // 2)
        #BC = BC * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1) 
        
        dtA = torch.exp(dtA.unsqueeze(-1) * torch.arange(L).to(dtA))
        h = 2 * torch.einsum("... d, ... d l -> ... l", C, dtA[None]).real

        h = h.view(1, self.d_model, L)
        return h, None


