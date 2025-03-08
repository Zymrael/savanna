import math
import torch
import torch.nn as nn

from einops import rearrange, repeat


has_pykeops = False
from savanna.ops.vandermonde import log_vandermonde_naive as log_vandermonde


_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class LaughingHyenaModalFilter(nn.Module):
    def __init__(self, global_config, d_model, order, theta_max):
        super().__init__()
        self.order = order
        self.d_model = d_model

        # Init poles and residues
        self.register_parameter("r", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("theta", nn.Parameter(torch.ones(order // 2, d_model)))
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


# class ParallelS4D(OptimModule):
#     """Version using (complex) diagonal state matrix (S4D)"""

#     def __init__(
#         self,
#         A,
#         B,
#         C,
#         log_dt,
#         L=None,
#         disc="bilinear",
#         real_type="exp",
#         lr=None,
#         bandlimit=None,
#         force_real=False,
#     ):
#         super().__init__()
#         self.L = L
#         self.disc = disc
#         self.bandlimit = bandlimit
#         self.real_type = real_type
#         self.force_real = force_real

#         # Rank of low-rank correction
#         assert A.size(-1) == C.size(-1)
#         self.H = log_dt.size(-1)
#         self.N = A.size(-1)
#         assert A.size(-2) == B.size(-2)  # Number of independent SSMs trained
#         assert self.H % A.size(-2) == 0
#         self.n_ssm = A.size(-2)
#         self.repeat = self.H // A.size(0)

#         self.channels = C.shape[0]
#         self.C = nn.Parameter(_c2r(_resolve_conj(C)))

#         # Register parameters
#         if lr is None or isinstance(lr, float):
#             lr_dict = {}
#         else:
#             lr_dict, lr = lr, None

#         self.register("log_dt", log_dt, lr_dict.get("dt", lr))
#         self.register("B", _c2r(B), lr_dict.get("B", lr))
#         self.register("inv_A_real", self._A_init(A.real), lr_dict.get("A", lr))
#         self.register("A_imag", A.imag, lr_dict.get("A", lr))

#     def _A_init(self, A_real):
#         A_real = torch.clamp(A_real, max=-1e-4)
#         if self.real_type == "none":
#             return -A_real
#         elif self.real_type == "exp":
#             return torch.log(-A_real)  # Some of the HiPPO methods have real part 0
#         elif self.real_type == "relu":
#             return -A_real
#         elif self.real_type == "sigmoid":
#             return torch.logit(-A_real)
#         elif self.real_type == "softplus":
#             return torch.log(torch.exp(-A_real) - 1)
#         else:
#             raise NotImplementedError

#     def _A(self):
#         # Get the internal A (diagonal) parameter
#         if self.real_type == "none":
#             A_real = -self.inv_A_real
#         elif self.real_type == "exp":
#             A_real = -torch.exp(self.inv_A_real)
#         elif self.real_type == "relu":
#             # JAX version seems to NaN if you alloA 0's, although this code Aas fine Aithout it
#             A_real = -F.relu(self.inv_A_real) - 1e-4
#         elif self.real_type == "sigmoid":
#             A_real = -F.sigmoid(self.inv_A_real)
#         elif self.real_type == "softplus":
#             A_real = -F.softplus(self.inv_A_real)
#         else:
#             raise NotImplementedError
#         A = A_real + 1j * self.A_imag
#         return A

#     def forward(self, L, state=None, u=None, **kwargs):
#         """
#         state: (B, H, N) initial state
#         rate: sampling rate factor
#         L: target length
#         returns:
#         (C, H, L) convolution kernel (generally C=1)
#         (B, H, L) output from initial state
#         """
#         rate = 1
#         log_dt = self.log_dt.to(torch.float32)
#         dt = torch.exp(log_dt) * rate  # (H)
#         C = _r2c(self.C.to(torch.float32))  # (C H N)
#         A = self._A()  # (H N)

#         B = _r2c(self.B.to(torch.float32))
#         B = repeat(B, "t n -> 1 (v t) n", v=self.repeat)

#         # Force A to be real valued, so the whole kernel can be interpreted as a "multi-head EMA"
#         if self.force_real:
#             A = A.real + 0j

#         if self.bandlimit is not None:
#             freqs = dt[:, None] / rate * A.imag.abs() / (2 * math.pi)  # (H, N)
#             mask = torch.where(freqs < self.bandlimit * 0.5, 1, 0)
#             C = C * mask

#         # Incorporate dt into A
#         A = repeat(A, "t n -> (v t) n", v=self.repeat)
#         dtA = A * dt.unsqueeze(-1)  # (H N)

#         # Augment B with state
#         if state is not None:
#             s = state / dt.unsqueeze(-1)
#             if self.disc == "bilinear":
#                 s = s * (1.0 + dtA / 2)
#             elif self.disc == "zoh":
#                 s = s * dtA * dtA.exp() / (dtA.exp() - 1.0)
#             B = torch.cat([s, B], dim=-3)  # (1+B H N)

#         C = (B[:, None, :, :] * C).view(-1, self.H, self.N)
#         if self.disc == "zoh":
#             # Power up
#             C = C * (torch.exp(dtA) - 1.0) / A
#             # TODO (TD): make it work for C.shape[0] > 1
#             if log_vandermonde_fast is not None and C.shape[0] == 1:
#                 K = log_vandermonde_fast(C.squeeze(0), dtA, L).unsqueeze(0)  # (H L)
#             else:
#                 K = log_vandermonde(C, dtA, L)  # (H L)
#         elif self.disc == "bilinear":
#             C = C * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A
#             dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
#             if log_vandermonde_fast is not None:
#                 dA_log = repeat(dA.log(), "h d -> (c h) d", c=C.shape[0])
#                 K = rearrange(
#                     log_vandermonde_fast(rearrange(C, "c h d -> (c h) d"), dA_log, L),
#                     "(c h) d -> c h d",
#                     c=C.shape[0],
#                 )
#             else:
#                 K = log_vandermonde(C, dA.log(), L)
#         elif self.disc == "dss":
#             # Implementation from DSS meant for case when real eigenvalues can be positive
#             P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device)  # [H N L]
#             A_gt_0 = A.real > 0  # [N]
#             if A_gt_0.any():
#                 with torch.no_grad():
#                     P_max = dtA * (A_gt_0 * (L - 1))  # [H N]
#                 P = P - P_max.unsqueeze(-1)  # [H N L]
#             S = P.exp()  # [H N L]

#             dtA_neg = dtA * (1 - 2 * A_gt_0)  # [H N]
#             num = dtA_neg.exp() - 1  # [H N]
#             den = (dtA_neg * L).exp() - 1  # [H N]

#             # Inline reciprocal function for DSS logic
#             x = den * A
#             x_conj = _resolve_conj(x)
#             r = x_conj / (x * x_conj + 1e-7)

#             C = C * num * r  # [C H N]
#             K = contract("chn,hnl->chl", C, S).float()
#             # [MP] Assume c=1, h=D case
#             K = rearrange(K, "c h l -> (c h) l")
#         else:
#             assert False, f"{self.disc} not supported"

#         K = K.view(-1, self.channels, self.H, L)  # (1+B C H L)
#         if state is not None:
#             K_state = K[:-1, :, :, :]  # (B C H L)
#         else:
#             K_state = None
#         K = K[-1, :, :, :]  # (C H L)
#         return K, K_state

#     def _setup_step(self):
#         # These methods are organized like this to be compatible with the NPLR kernel interface
#         dt = torch.exp(self.log_dt)  # (H)
#         B = _r2c(self.B)  # (H N)
#         C = _r2c(self.C)  # (C H N)
#         self.dC = C
#         A = self._A()  # (H N)

#         A = repeat(A, "t n -> (v t) n", v=self.repeat)
#         B = repeat(B, "t n -> (v t) n", v=self.repeat)

#         # Incorporate dt into A
#         dtA = A * dt.unsqueeze(-1)  # (H N)
#         if self.disc == "zoh":
#             self.dA = torch.exp(dtA)  # (H N)
#             self.dB = B * (torch.exp(dtA) - 1.0) / A  # (C H N)
#         elif self.disc == "bilinear":
#             self.dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
#             self.dB = B * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A

#     def default_state(self, *batch_shape):
#         C = _r2c(self.C)
#         state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
#         return state

#     def step(self, u, state):
#         next_state = contract("h n, b h n -> b h n", self.dA, state) + contract(
#             "h n, b h -> b h n", self.dB, u
#         )
#         y = contract("c h n, b h n -> b c h", self.dC, next_state)
#         return 2 * y.real, next_state

#     def forward_state(self, u, state):
#         self._setup_step()
#         AL = self.dA ** u.size(-1)
#         u = u.flip(-1).to(self.dA).contiguous()  # (B H L)
#         v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
#         next_state = AL * state + v
#         return next_state


class ImplicitComplexModalFilter(OptimModule):

    def __init__(
        self,
        A,
        B,
        C,
        log_dt,
        L=None,
        disc="bilinear",
        real_type="exp",
        lr=None,
        bandlimit=None,
    ):
        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type

        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2)
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr_dict.get("dt", lr))
        self.register("B", _c2r(B), lr_dict.get("B", lr))
        self.register("inv_A_real", torch.log(-A.real), lr_dict.get("A", lr))
        self.register("A_imag", A.imag.contiguous(), lr_dict.get("A", lr))

    def forward(self, L, state=None, u=None, **kwargs):
        log_dt = self.log_dt.to(torch.float32)
        dt = torch.exp(log_dt)
        C = _r2c(self.C.to(torch.float32))  # (C H N)

        A_real = -torch.exp(self.inv_A_real)
        A_imag = self.A_imag
        A = A_real + 1j * self.A_imag

        B = _r2c(self.B.to(torch.float32))
        B = repeat(B, "t n -> 1 (v t) n", v=self.repeat)

        A = repeat(A, "t n -> (v t) n", v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        C = C * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A
        dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
        dA_log = repeat(dA.log(), "h d -> (c h) d", c=C.shape[0])
        K = log_vandermonde(C, dtA, L)  # (H L)

        K = K.view(-1, self.channels, self.H, L)  # (1+B C H L)
        K = K[-1, :, :, :]  # (C H L)
        return K, None


class ParallelS5(OptimModule):
    def __init__(self):
        raise NotImplementedError


class ParallelLRU(OptimModule):
    def __init__(self):
        raise NotImplementedError


class ParallelModalLH(OptimModule):
    def __init__(self):
        raise NotImplementedError
