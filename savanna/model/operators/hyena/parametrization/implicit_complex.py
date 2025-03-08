
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from opt_einsum import contract

from savanna.model.operators.hyena.parametrization.utils import combination
from savanna.model.operators.hyena.parametrization.implicit_complex_inner import ImplicitComplexModalFilter
from savanna.utils import ALLOC_DEVICE

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)


@torch.no_grad()
def power(L, A, v=None):
    """Compute A^L and the scan sum_i A^i v_i
    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A)  # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I
        L //= 2
        if L == 0:
            break
        l *= 2
        if v is None:
            powers = [powers[-1] @ powers[-1]]
        else:
            powers.append(powers[-1] @ powers[-1])

    if v is None:
        return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, "... (z l) -> ... z l", z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


class ParallelComplexModalFilter(nn.Module):
    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="diag-lin",
        rank=1,
        channels=1,
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        lr=None,
        mode="diag",
        n_ssm=None,
        verbose=False,
        measure_args={},
        **kernel_args,
    ):
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min, device=ALLOC_DEVICE), math.log(dt_max), H, device=ALLOC_DEVICE))
        else:
            log_dt = torch.rand(self.H, dtype=dtype, device=ALLOC_DEVICE) * (math.log(dt_max) - math.log(dt_min)) + math.log(
                dt_min
            )

        w, P, B, V = combination(measure, self.N, rank, self.n_ssm, **measure_args)

        if deterministic:
            C = torch.zeros(channels, self.n_ssm, self.N, dtype=cdtype, device=ALLOC_DEVICE)
            C[:, :, :1] = 1.0
            C = contract("hmn, chn -> chm", V.conj().transpose(-1, -2), C)  # V^* C
            C = repeat(C, "c t n -> c (v t) n", v=self.n_ssm // C.size(-2)).clone().contiguous()
        else:
            C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype, device=ALLOC_DEVICE)

        assert self.n_ssm % B.size(-2) == 0 and self.n_ssm % P.size(-2) == 0 and self.n_ssm % w.size(-2) == 0
        B = repeat(B, "t n -> (v t) n", v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, "r t n -> r (v t) n", v=self.n_ssm // P.size(-2)).clone().contiguous()
        w = repeat(w, "t n -> (v t) n", v=self.n_ssm // w.size(-2)).clone().contiguous()

        if mode == "diag":
            C = C * repeat(B, "t n -> (v t) n", v=H // self.n_ssm)
            self.kernel = ImplicitComplexModalFilter(
                w,
                B,
                C,
                log_dt,
                L=L,
                lr=lr,
                **kernel_args,
            )
        else:
            raise NotImplementedError(f"{mode=} is not valid")

    def forward(self, L=None, state=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)
