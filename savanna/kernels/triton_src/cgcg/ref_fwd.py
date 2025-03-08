"""
This module implements the Conv-Gate-Conv-Gate (CGCG) and Gate-Conv-Gate (GCG) variants.

Notation used throughout the module:
    bs (int): batch size
    l (int): sequence length
    g (int): number of groups in channels
    dg (int): group dimension
    hl (int): filter length (ell_h) in the notes
    dgl (int): size of length groups
    gl (int): number of groups
"""

import torch
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from einops import rearrange

from .utils import correction_toeplitz, toeplitz


def gcg_fwd_ref_original(x, B, C, h, use_causal_conv=False):
    bs, l, g, dg = x.shape
    hl = h.shape[-1]
    if hl > 3:
        use_causal_conv = False
    d = g * dg

    Bx = B * x
    Bx_l_last = Bx.permute(0, 2, 3, 1)  # b, g, dg, l

    # here we assume some weight sharing structure between groups in filters,
    Bx_l_last_flattened = Bx_l_last.reshape(bs, -1, l)  # b, d, l
    h_grouped = h.repeat(g, 1, 1)  # d, hl, 1

    # reminder: no flip on the filter
    if use_causal_conv:
        # only supports g = 1 for now, general support requires more reshapes
        y_l_last_flattened = causal_conv1d_fn(Bx_l_last_flattened, h_grouped[:, 0])
    else:
        y_l_last_flattened = F.conv1d(
            Bx_l_last_flattened, h_grouped, groups=d, stride=1, padding=hl - 1
        )[..., : -hl + 1]

    y_l_last = y_l_last_flattened.reshape(bs, g, dg, l)
    y = y_l_last.permute(0, 3, 1, 2)
    return C * y


def gcg_two_pass_chunked_fwd_original(x, B, C, h, gl):
    "Local plus correction approach."
    l = x.shape[1]
    hl = h.shape[0]
    chunk_size = gl

    num_l_chunks = l // chunk_size
    h = h.flip(-1)

    # first step
    Bx = B * x
    Bx = rearrange(
        Bx, "b (gl dgl) g dg -> b g gl dg dgl", gl=num_l_chunks, dgl=chunk_size
    )
    T_local = toeplitz(h[:, 0], chunk_size).to(x.device)

    y = torch.einsum("bgldi,dji->bgldj", Bx, T_local)

    # second step: correction
    T_hat = correction_toeplitz(h[:, 0], chunk_size).to(x.device)
    Bx = torch.roll(Bx, 1, dims=2)
    Bx[:, :, 0] = 0
    y = y + torch.einsum("bgldi,dji->bgldj", Bx, T_hat)
    y = rearrange(y, "b g gl dg dgl -> b (gl dgl) g dg")
    return C * y


def gcg_fwd_ref_corrected(x, B, C, h, use_causal_conv=False, interleave=True, return_intermediates=False):
    """
    This is same as the original fwd_ref except how we interpret `g` and `dg`

    Crucially, `h` has shape (g, 1, hl), where
    `g` = number of feature channel groups and `dg` = number of channels per group

    `g` is thus the number of distinct filters

    E.g.:
    ```python
        h = torch.randn(g, 1, hl)
    ```

    This is done to be consistent with cgcg.triton.fwd_kernels.two_pass_fwd_grouped.
    
    Also, the filters are grouped in a blocked arrangement along the feature dimension using `repeat_interleave`
    whereas in the original implementation, the filters were grouped using `repeat`, which stripes
    the filters across feature dimensions.

    """
    bs, l, g, dg = x.shape
    hl = h.shape[-1]
    if hl > 3:
        use_causal_conv = False
    d = g * dg

    Bx = B * x
    Bx_l_last = Bx.permute(0, 2, 3, 1)  # b, g, dg, l

    # here we assume some weight sharing structure between groups in filters,
    Bx_l_last_flattened = Bx_l_last.reshape(bs, -1, l)  # b, d, l
    if interleave:
        h_grouped = h.repeat_interleave(dg, dim=0)  # d, 1, hl
    else:
        h_grouped = h.repeat(dg, 1, 1)  # d, 1, hl

    if use_causal_conv:
        # only supports g = 1 for now, general support requires more reshapes
        y_l_last_flattened = causal_conv1d_fn(Bx_l_last_flattened, h_grouped[:, 0])
    else:
        y_l_last_flattened = F.conv1d(
            Bx_l_last_flattened, h_grouped, groups=d, stride=1, padding=hl - 1
        )[..., : -hl + 1]

    y_l_last = y_l_last_flattened.reshape(bs, g, dg, l)
    y2 = y_l_last.permute(0, 3, 1, 2)
    y = C * y2
    
    if return_intermediates:
        return (Bx, # pre-gate output
                y2, # Result of convolution
                y # post-gate output
                )
    return y
def gcg_fwd_ref_corrected_noncausal(x, B, C, h, return_intermediates=False):
    """
    This is same as the original fwd_ref except how we interpret `g` and `dg`

    Crucially, `h` has shape (g, 1, hl), where
    `g` = number of feature channel groups and `dg` = number of channels per group

    `g` is thus the number of distinct filters

    E.g.:
    ```python
        h = torch.randn(g, 1, hl)
    ```

    This is done to be consistent with cgcg.triton.fwd_kernels.two_pass_fwd_grouped.
    
    Also, the filters are grouped in a blocked arrangement along the feature dimension using `repeat_interleave`
    whereas in the original implementation, the filters were grouped using `repeat`, which stripes
    the filters across feature dimensions.

    """
    bs, l, g, dg = x.shape
    hl = h.shape[-1]
    d = g * dg

    Bx = B * x
    Bx_l_last = Bx.permute(0, 2, 3, 1)  # b, g, dg, l

    # here we assume some weight sharing structure between groups in filters,
    Bx_l_last_flattened = Bx_l_last.reshape(bs, -1, l)  # b, d, l
    
    h_grouped = h.repeat_interleave(dg, dim=0)  # d, 1, hl
    

    y_l_last_flattened = F.conv1d(
            Bx_l_last_flattened, h_grouped, groups=d, stride=1, padding=hl - 1
        )[..., : -hl + 1]

    y_l_last = y_l_last_flattened.reshape(bs, g, dg, l)
    y = y_l_last.permute(0, 3, 1, 2)
    
    if return_intermediates:
        return Bx_l_last_flattened, h_grouped, y, C*y
    
    return C * y

def gcg_fwd_ref_corrected_causal(x, B, C, h):
    """
    This is same as the original fwd_ref except how we interpret `g` and `dg`

    Crucially, `h` has shape (g, 1, hl), where
    `g` = number of feature channel groups and `dg` = number of channels per group

    `g` is thus the number of distinct filters

    E.g.:
    ```python
        h = torch.randn(g, 1, hl)
    ```

    This is done to be consistent with cgcg.triton.fwd_kernels.two_pass_fwd_grouped.
    
    Also, the filters are grouped in a blocked arrangement along the feature dimension using `repeat_interleave`
    whereas in the original implementation, the filters were grouped using `repeat`, which stripes
    the filters across feature dimensions.

    """
    bs, l, g, dg = x.shape
    hl = h.shape[-1]
    assert hl < 3
    
    d = g * dg

    Bx = B * x
    Bx_l_last = Bx.permute(0, 2, 3, 1)  # b, g, dg, l

    # here we assume some weight sharing structure between groups in filters,
    Bx_l_last_flattened = Bx_l_last.reshape(bs, -1, l)  # b, d, l
    
    h_grouped = h.repeat_interleave(dg, dim=0)  # d, 1, hl
    
    y_l_last_flattened = causal_conv1d_fn(Bx_l_last_flattened, h_grouped[:, 0])

    y_l_last = y_l_last_flattened.reshape(bs, g, dg, l)
    y = y_l_last.permute(0, 3, 1, 2)
    return C * y
def gcg_two_pass_chunked_fwd_corrected(x, B, C, h, gl, return_intermediates=False):
    """
    The original gcg_two_pass_chunked_fwd function does not work for g > 1
    This version works for g = 1 and is implemented with torch-native ops (no einops)
    """

    b, l, g, dg = x.shape
    chunk_size = gl

    num_l_chunks = l // chunk_size
    h = h.flip(-1)
    # first step
    Bx = B * x

    Bx = Bx.reshape(b, num_l_chunks, chunk_size, g, dg)
    # b gl g dgl dg
    Bx = Bx.permute(0, 1, 3, 2, 4)
    T_local = toeplitz(h[:, 0], chunk_size, dtype=x.dtype, device=x.device)

    # T_local is g x chunk_size x chunk_size -> g x i x j, need to dot product along j
    # j maps to dgl
    assert T_local.shape == (g, chunk_size, chunk_size)
    # (M x K) x (K x N), M -> i (chunk_size), K -> j (chunk_size), N -> d (dg)
    y = torch.matmul(T_local, Bx)

    if return_intermediates:
        y1 = y.detach().clone()
        # b gl dgl g dg
        y1 = y1.permute(0, 1, 3, 2, 4)
        y1 = y1.reshape(b, l, g, dg)

    # second step: correction
    T_hat = correction_toeplitz(h[:, 0], chunk_size, dtype=x.dtype, device=x.device)

    # Bx is b gl g dgl dg, need to set first chunk to 0
    Bx = torch.roll(Bx, 1, dims=1)
    Bx[:, 0, :] = 0

    if return_intermediates:
        Bx_lag = Bx.detach().clone()
        Bx_lag = Bx_lag.permute(0, 1, 3, 2, 4).reshape(b, l, g, dg)

    correction_term = torch.matmul(T_hat, Bx)

    y = y + correction_term
    y = y.permute(0, 1, 3, 2, 4).reshape(b, l, g, dg)
    if return_intermediates:
        correction_term = correction_term.permute(0, 1, 3, 2, 4).reshape(b, l, g, dg)
        return (y1, 
                Bx_lag, 
                correction_term, 
                y, # y2 output of convolution reshaped
                C * y) # y final
    else:
        return C * y