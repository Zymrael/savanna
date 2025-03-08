import torch
from .utils import correction_toeplitz, toeplitz


def dh_local(dT_local, g, hl, chunk_size):
    """
    Calculates partial dh given dT_local
    """

    dT = dT_local.reshape(g, chunk_size, chunk_size)
    dh = []
    for i in reversed(range(hl)):
        dhdi = dT.diagonal(offset=-i, dim1=1, dim2=2).sum(-1, keepdim=True)
        dh.append(dhdi)
    return torch.cat(dh, 1).reshape(g, 1, hl)


def dh_correction(dT_correction, g, hl, chunk_size):
    """
    Calculates partial dh given dT_correction
    """
    dTc = dT_correction.reshape(g, chunk_size, chunk_size)
    dh = []
    for i in range(hl):
        offset = chunk_size - (hl - 1) + i
        dhdi = dTc.diagonal(offset=offset, dim1=1, dim2=2).sum(-1, keepdim=True)
        dh.append(dhdi)
    return torch.cat(dh, 1).reshape(g, 1, hl)


def gcg_two_pass_chunked_bwd(dy, x, y2, B, C, h, gl, return_dT=False):
    """
    Reference two pass backward pass.

    Args:
        dy: (bs, l, g, dg)
        x: (bs, l, g, dg)
        y2: (bs, l, g, dg) = T_local @ B*x + T_correction @ B*x, assume saved from forward pass
        B: (bs, l, g, dg)
        C: (bs, l, g, dg)
        h: (g, 1, filter_size)
        gl: int
    Notes:
    - toeplitz matrices need to be transposed
    - lagging matrix needs to be rolled in the opposite direction, since the
    gradients need to be backpropped to t-1 terms

    """
    b, l, g, dg = dy.shape
    chunk_size = gl
    hl = h.shape[-1]
    num_l_chunks = l // chunk_size
    h_ = h.flip(-1)

    # Start backprop
    dC = dy * y2

    # Backprop through C
    dy = dy * C

    # Chunked computations
    # Calculate convolution and correction terms separately for ease of explication
    dy = dy.reshape(b, num_l_chunks, chunk_size, g, dg).permute(0, 1, 3, 2, 4)

    # First convolution grad
    T_local = toeplitz(h_[:, 0], chunk_size).to(dy.device).to(dy.dtype)
    # Permute before matmul
    T_local = T_local.permute(0, 2, 1)
    dy1 = torch.matmul(T_local, dy)
    dy1 = dy1.permute(0, 1, 3, 2, 4).reshape(b, l, g, dg)
    dy1dx = dy1 * B
    dy1dB = dy1 * x

    # Calculate gradient wrt h_local
    Bx = B * x
    Bx_reshaped = (Bx).reshape(b, num_l_chunks, chunk_size, g, dg).permute(0, 1, 3, 2, 4)
    Bx_reshaped_transposed = Bx_reshaped.permute(0, 1, 2, 4, 3)
    dT_local = torch.matmul(dy, Bx_reshaped_transposed)

    # Sum over batch and sequence dimensions
    dT_local = dT_local.sum(dim=(0, 1))
    dhl = dh_local(dT_local, g, hl, chunk_size)

    # Correction grad
    T_hat = correction_toeplitz(h_[:, 0], chunk_size).to(dy.device).to(dy.dtype)
    T_hat = T_hat.permute(0, 2, 1)

    # Note the direction of roll from lagged to leading
    dy_lag = torch.roll(dy, -1, dims=1)
    dy_lag[:, -1, :] = 0

    dcorrection = torch.matmul(T_hat, dy_lag)
    dcorrection = dcorrection.permute(0, 1, 3, 2, 4).reshape(b, l, g, dg)
    dcorrectiondx = dcorrection * B
    dorrectiondB = dcorrection * x

    # Calculate grad wrt h_correction
    Bx_lag = torch.roll(Bx_reshaped, 1, dims=1)
    Bx_lag[:, 0, :] = 0
    Bx_lag_transposed = Bx_lag.permute(0, 1, 2, 4, 3)
    dT_correction = torch.matmul(dy, Bx_lag_transposed)
    # Sum over batch and sequence dimensions
    dT_correction = dT_correction.sum(dim=(0, 1))
    dhc = dh_correction(dT_correction, g, hl, chunk_size)

    # Combine conv and correction gradients
    dx = dy1dx + dcorrectiondx
    dB = dy1dB + dorrectiondB
    dh = dhl + dhc

    grads = [dx, dB, dC, dh]
    if return_dT:
        grads.extend([dT_local, dT_correction])

    return grads
