"""Initializations of structured state space models"""

import math
import torch
import numpy as np
from scipy import special as ss
from einops import rearrange, repeat
from opt_einsum import contract
from savanna.utils import ALLOC_DEVICE


def get_dtype_from_string(dtype_str):
    if type(dtype_str) == str:
        if dtype_str == "float32" or dtype_str == "fp32":
            return torch.float32
        elif dtype_str == "float16" or dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bfloat16" or dtype_str == "bf16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unrecognized dtype {dtype_str}")
    else:
        return dtype_str


def embed_c2r(A):
    A = rearrange(A, "... m n -> ... m () n ()")
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + np.pad(A, ((0, 0), (1, 0), (0, 0), (1, 0)))
    return rearrange(A, "m x n y -> (m x) (n y)")


# TODO take in 'torch' option to return torch instead of numpy, and converts the shape of B from (N, 1) to (N)
def transition(measure, N, **measure_args):
    """A, B transition matrices for different measures
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == "lagt":
        b = measure_args.get("beta", 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == "glagt":
        alpha = measure_args.get("alpha", 0.0)
        beta = measure_args.get("beta", 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(0.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1)))
        A = (1.0 / L[:, None]) * A * L[None, :]
        B = (1.0 / L[:, None]) * B * np.exp(-0.5 * ss.gammaln(1 - alpha)) * beta ** ((1 - alpha) / 2)
    # Legendre (translated)
    elif measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # LMU: equivalent to LegT up to normalization
    elif measure == "lmu":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
    # Legendre (scaled)
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == "legsd":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += 0.5 * B * B[None, :, 0]
        B = B / 2.0
    elif measure in ["fourier_diag", "foud"]:
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        A = A - 0.5 * np.eye(N)
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1
        B = B[:, None]
    elif measure in ["fourier", "fout"]:
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    elif measure == "fourier_decay":
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - 0.5 * B[:, None] * B[None, :]
        B = 0.5 * B[:, None]
    elif measure == "fourier2":  # Double everything: orthonormal on [0, 1]
        freqs = 2 * np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :] * 2
        B = B[:, None] * 2
    elif measure == "random":
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == "diagonal":
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal"""

    if measure == "legs":
        assert rank >= 1
        P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype, device=ALLOC_DEVICE)).unsqueeze(0)  # (1 N)
    elif measure == "legt":
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype, device=ALLOC_DEVICE))  # (N)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2 N)
        P *= 2 ** (-0.5)  # Halve the rank correct just like the original matrix was halved
    elif measure == "lagt":
        assert rank >= 1
        P = 0.5**0.5 * torch.ones(1, N, dtype=dtype, device=ALLOC_DEVICE)
    elif measure in ["fourier", "fout"]:
        P = torch.zeros(N, device=ALLOC_DEVICE)
        P[0::2] = 2**0.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure == "fourier_decay":
        P = torch.zeros(N, device=ALLOC_DEVICE)
        P[0::2] = 2**0.5
        P[0] = 1
        P = P.unsqueeze(0)
        P = P / 2**0.5
    elif measure == "fourier2":
        P = torch.zeros(N, device=ALLOC_DEVICE)
        P[0::2] = 2**0.5
        P[0] = 1
        P = 2**0.5 * P.unsqueeze(0)
    elif measure in ["fourier_diag", "foud", "legsd"]:
        P = torch.zeros(1, N, dtype=dtype, device=ALLOC_DEVICE)
    else:
        raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype, device=ALLOC_DEVICE)], dim=0)  # (rank N)
    return P


def initial_C(measure, N, dtype=torch.float):
    """Return C that captures the other endpoint in the HiPPO approximation"""

    if measure == "legt":
        C = (torch.arange(N, dtype=dtype, device=ALLOC_DEVICE) * 2 + 1) ** 0.5 * (-1) ** torch.arange(N, device=ALLOC_DEVICE)
    elif measure == "fourier":
        C = torch.zeros(N, device=ALLOC_DEVICE)
        C[0::2] = 2**0.5
        C[0] = 1
    else:
        C = torch.zeros(N, dtype=dtype, device=ALLOC_DEVICE)  # (N)

    return C


def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype, device=ALLOC_DEVICE)  # (N, N)
    B = torch.as_tensor(B, dtype=dtype, device=ALLOC_DEVICE)[:, 0]  # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)  # (r N)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (
        err := torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N
    ) > 1e-5:  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)

    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision:
        AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    w_im, V = torch.linalg.eigh(AP * -1j)  # (..., N) (..., N, N)
    if diagonalize_precision:
        w_im, V = w_im.to(cdtype), V.to(cdtype)
    w = w_re + 1j * w_im
    # Check: V w V^{-1} = A
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    # Only keep half of each conjugate pair
    _, idx = torch.sort(w.imag)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, : N // 2]
    w = w_sorted[: N // 2]
    assert w[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if w[-1].abs() < 1e-4:
        V[:, -1] = 0.0
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
    if (err := torch.sum((2 * _AP.real - AP) ** 2) / N) > 1e-5:
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    # C = initial_C(measure, N, dtype=dtype)
    B = contract("ij, j -> i", V_inv, B.to(V))  # V^* B
    # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = contract("ij, ...j -> ...i", V_inv, P.to(V))  # V^* P

    # return w, P, B, C, V
    return w, P, B, V


def dplr(
    scaling="linear",
    N=64,
    rank=1,
    H=1,
    dtype=torch.float,
    real_scale=1.0,
    imag_scale=1.0,
    random_real=False,
    random_imag=False,
    normalize=False,
    diagonal=True,
    random_B=False,
):
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.full((), math.pi, device=ALLOC_DEVICE)
    if random_real:
        real_part = torch.rand(H, N // 2, device=ALLOC_DEVICE)
    else:
        real_part = 0.5 * torch.ones(H, N // 2, device=ALLOC_DEVICE)
    if random_imag:
        imag_part = N // 2 * torch.rand(H, N // 2, device=ALLOC_DEVICE)
    else:
        imag_part = repeat(torch.arange(N // 2, device=ALLOC_DEVICE), "n -> h n", h=H)

    real_part = real_scale * real_part
    if scaling == "random":
        imag_part = torch.randn(H, N // 2, device=ALLOC_DEVICE)
    elif scaling == "real":
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N // 2, device=ALLOC_DEVICE), "n -> h n", h=H)
    elif scaling in ["linear", "lin"]:
        imag_part = pi * imag_part
    elif scaling in [
        "inverse",
        "inv",
    ]:  # Based on asymptotics of the default HiPPO matrix
        imag_part = 1 / pi * N * (N / (1 + 2 * imag_part) - 1)
    elif scaling in ["inverse2", "inv2"]:
        imag_part = 1 / pi * N * (N / (1 + imag_part) - 1)
    elif scaling in ["quadratic", "quad"]:
        imag_part = 1 / pi * (1 + 2 * imag_part) ** 2
    elif scaling in ["legs", "hippo"]:
        w, _, _, _ = nplr("legsd", N)
        imag_part = w.imag

    else:
        raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part

    # Initialize B
    if random_B:
        B = torch.randn(H, N // 2, dtype=dtype, device=ALLOC_DEVICE)
    else:
        B = torch.ones(H, N // 2, dtype=dtype, device=ALLOC_DEVICE)

    if normalize:
        norm = -B / w  # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2 * torch.sum(torch.abs(norm) ** 2, dim=-1, keepdim=True)  # Variance with a random C vector
        B = B / zeta**0.5

    P = torch.randn(rank, H, N // 2, dtype=dtype, device=ALLOC_DEVICE)
    if diagonal:
        P = P * 0.0
    V = torch.eye(N, dtype=dtype, device=ALLOC_DEVICE)[:, : N // 2]  # Only used in testing
    V = repeat(V, "n m -> h n m", h=H)

    return w, P, B, V


def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization
    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if measure == "dplr":
        w, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    elif measure.startswith("diag"):
        args = measure.split("-")
        assert args[0] == "diag" and len(args) > 1
        scaling = args[1]
        w, P, B, V = dplr(scaling=scaling, N=N, rank=R, H=H, diagonal=True, **ssm_args)
    else:
        w, P, B, V = nplr(measure, N, R, **ssm_args)
        w = repeat(w, "n -> s n", s=H)
        P = repeat(P, "r n -> r s n", s=H)
        B = repeat(B, "n -> s n", s=H)
        V = repeat(V, "n m -> s n m", s=H)
    return w, P, B, V


combinations = {
    "hippo": ["legs", "fourier"],
    "diag": ["diag-inv", "diag-lin"],
    "all": ["legs", "fourier", "diag-inv", "diag-lin"],
}


def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]

    assert (
        S % len(measures) == 0
    ), f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    w, P, B, V = zip(*[ssm(measure, N, R, S // len(measures), **ssm_args) for measure in measures])
    w = torch.cat(w, dim=0)  # (S N)
    P = torch.cat(P, dim=1)  # (R S N)
    B = torch.cat(B, dim=0)  # (S N)
    V = torch.cat(V, dim=0)  # (S N N)
    return w, P, B, V
