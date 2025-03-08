import itertools

import torch

from savanna.memory_stats import (
    _MEMORY_PARENT_KEY,
    _make_alloc_keys,
    _make_count_keys,
    get_memory_stats,
)


def make_tensor(size=1):
    """
    Create tensor of specified size in MB
    """
    return torch.ones(size * 1024 * 1024, dtype=torch.uint8, device="cuda")


def test_mem_stats():

    # Allocated 1MB
    t = make_tensor(1)
    # Returns memory stats as flattened dictionary separated by '/' per wandb format
    mem_stats = get_memory_stats()
    assert all(k.startswith(_MEMORY_PARENT_KEY) for k in mem_stats.keys())
    alloc_keys = _make_alloc_keys()
    count_keys = _make_count_keys()
    assert all(k in mem_stats for k in alloc_keys)
    assert all(k in mem_stats for k in count_keys)


if __name__ == "__main__":
    test_mem_stats()
