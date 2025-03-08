import itertools
import pytest
import torch

from savanna.kernels.triton_src.cgcg.utils import toeplitz

from .utils import print_diff, toeplitz_load_kernel

torch.manual_seed(0)
torch.set_default_device("cuda")


class ToeplitzTestConfig:
    DEFAULT_CHUNK_SIZES = [16, 32, 128]
    DEFAULT_GROUP_SIZES = [1, 2]
    DEFAULT_FILTER_LENS = [4, 128]

    @staticmethod
    def configs(chunk_sizes: list[int] = None, group_sizes: list[int] = None, filter_lens: list[int] = None):
        if chunk_sizes is None:
            chunk_sizes = ToeplitzTestConfig.DEFAULT_CHUNK_SIZES
        if group_sizes is None:
            group_sizes = ToeplitzTestConfig.DEFAULT_GROUP_SIZES
        if filter_lens is None:
            filter_lens = ToeplitzTestConfig.DEFAULT_FILTER_LENS
        return list(itertools.product(chunk_sizes, group_sizes, filter_lens))


@pytest.mark.parametrize("g, hl, chunk_size", ToeplitzTestConfig.configs())
def test_toeplitz_load(g, hl, chunk_size):
    # Ensure that filter length is less than or equal to chunk size
    chunk_size = max(hl, chunk_size)

    # Compile-time constant for triton kernel
    is_single_group = g == 1

    # Initialize filter
    # `g` is the number of groups (number of distinct filters), `hl` is the filter length
    h = torch.arange(g * 1 * hl).reshape(g, 1, hl)

    # Reference toeplitz
    h_ = h[:, 0].flip(-1)
    T_ref = toeplitz(h_, chunk_size)

    # Kernel loads single filter at a time
    for i in range(g):
        ref = T_ref[i]
        T = torch.empty_like(ref)
        toeplitz_load_kernel[(1,)](
            h.reshape(-1), i, T, FILTER_LEN=hl, CHUNK_SIZE=chunk_size, SINGLE_GROUP=is_single_group
        )
        passed = torch.allclose(T, ref)
        if not passed:
            print(f"T_ref:\n{ref}")
            print(f"T:\n{T}")
            print_diff(T, ref)
        assert passed
