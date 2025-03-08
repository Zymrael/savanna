import torch
import torch.nn.functional as F
from einops import rearrange


@torch.jit.script
def _mul_sum(y, q):
    return (y * q).sum(dim=1)


def fftconv_func(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    # check if k is less than seqlen
    if k.shape[-1] < seqlen:
        # Pad the filter k to the length of the input sequence u
        k_padded = torch.nn.functional.pad(k, (0, seqlen - k.shape[-1]))

    # bidirectional
    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

        # split k along the channel dimension
        k, k2 = k.split(k.shape[1] // 2, dim=1)

        # get fft of both k's
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size

        if len(u.shape) > 3:
            k_f = k_f.unsqueeze(1)
            k2_f = k2_f.unsqueeze(1)

        y1 = u_f * k_f
        y2 = u_f.conj() * k2_f.conj()

        y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]

    # causal
    else:
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
