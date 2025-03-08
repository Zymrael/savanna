import torch
import torch.utils.benchmark as benchmark


def toeplitz(h, size, dtype=None, device=None):
    """
    Args:
    h (torch.Tensor): filter
    size (int): Size of the output Toeplitz matrix

    Returns:
    torch.Tensor: Toeplitz matrix
    """
    if dtype is None:
        dtype = h.dtype
    if device is None:
        device = h.device

    g = h.shape[0]
    T = torch.zeros(g, size, size, device=device, dtype=dtype)
    for i in range(size):
        for j in range(size):
            if i >= j and i - j < h.shape[1]:
                T[:, i, j] = h[:, i - j]
    return T


def toeplitz_from_empty(T, h):
    "Fills a device tensor with the Toeplitz matrix of the filter h"
    size = T.shape[1]
    for i in range(T.shape[1]):
        for j in range(T.shape[2]):
            if i >= j and i - j < h.shape[1]:
                T[:, i, j] = h[:, i - j]
    return T


def correction_toeplitz(h, size, dtype=None, device=None):
    "T_hat in the notes"
    if dtype is None:
        dtype = h.dtype
    if device is None:
        device = h.device

    g = h.shape[0]
    T = torch.zeros(g, size, size, device=device, dtype=dtype)
    for i in range(size):
        for j in range(size):
            if j >= i and i + size - j < h.shape[1]:
                T[:, i, j] = h[:, i + size - j]
    return T


def correction_toeplitz_from_empty(T, h):
    "Fills a device tensor with the correction Toeplitz matrix of the filter h"
    size = T.shape[1]
    for i in range(T.shape[1]):
        for j in range(T.shape[2]):
            if j >= i and i + size - j < h.shape[1]:
                T[:, i, j] = h[:, i + size - j]
    return T


def benchmark_forward(fn, *inputs, repeats=30, desc="", amp=False, amp_dtype=torch.float16, **kwinputs):
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m
