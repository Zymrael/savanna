import math
import torch.nn.functional as F
import torch
from einops import rearrange
import time


def fftconv_func(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv1d(x, k, bias):
    L = x.shape[-1]
    D = x.shape[1]
    return F.conv1d(x, k.flip(-1), groups=D, padding=L - 1)[..., :L]  # + bias * x


times = {}
times_fftconv = {}

D_list = [1024]
L_list = [2048, 4096, 8192]

for D in D_list:
    for L in L_list:
        x = torch.randn(32, D, L).to(device)
        k = torch.randn(D, 1, L).to(device)
        bias = torch.zeros(D).to(device)

        # conv1d in bf16, fftconv in fp32]
        x = x.to(dtype=torch.bfloat16)
        k = k.to(dtype=torch.bfloat16)
        bias = bias.to(dtype=torch.bfloat16)

        start = time.time()
        y = conv1d(x, k, bias)
        end = time.time()

        times[(D, L)] = end - start

        x = x.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        bias = bias.to(dtype=torch.float32)

        start = time.time()
        y2 = fftconv_func(x, k[:, 0], bias, None, False)
        end = time.time()

        times_fftconv[(D, L)] = end - start

        # print(y, y2)
        # assert torch.allclose(y, y2, rtol=1e-3, atol=1e-3)

print("conv1d")
print(times)

print("fftconv")
print(times_fftconv)
