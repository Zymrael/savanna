import triton
import triton.language as tl


@triton.jit
def toeplitz_idx(
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    TOEPLITZ_TYPE: tl.constexpr = "toeplitz",
):
    """
    Generates pointer indices relative to a base filter pointer for materializing toeplitz / correction toeplitz matrix
    directly on-chip.
    """
    if TOEPLITZ_TYPE == "toeplitz":
        r_idx = tl.arange((FILTER_LEN - 1), CHUNK_SIZE + (FILTER_LEN - 1))[None, :]
    elif TOEPLITZ_TYPE == "correction_toeplitz":
        r_idx = tl.arange((FILTER_LEN - 1), CHUNK_SIZE + (FILTER_LEN - 1))[None, :] - CHUNK_SIZE
    else:
        tl.static_assert(False, "Invalid ToeplitzType")
    c_idx = tl.arange(0, CHUNK_SIZE)[:, None]
    idx = r_idx - c_idx
    return idx


@triton.jit
def load_toeplitz(
    h_ptr,
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SINGLE_GROUP: tl.constexpr = True,
    group_num=0,
):
    t_idx = toeplitz_idx(FILTER_LEN, CHUNK_SIZE, "toeplitz")
    mask = (0 <= t_idx) & (t_idx < FILTER_LEN)

    if SINGLE_GROUP:
        T = tl.load(
            h_ptr + t_idx, mask=mask, other=0.0, eviction_policy="evict_last"
        )  # Want T to stay resident in L2 cache
    else:
        T = tl.load(
            h_ptr + group_num * FILTER_LEN + t_idx,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )

    return T


@triton.jit
def load_correction_toeplitz(
    h_ptr,
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SINGLE_GROUP: tl.constexpr = True,
    group_num=0,
):
    t_idx = toeplitz_idx(FILTER_LEN, CHUNK_SIZE, "correction_toeplitz")
    mask = (0 <= t_idx) & (t_idx < FILTER_LEN)

    if SINGLE_GROUP:
        T_C = tl.load(
            h_ptr + t_idx, mask=mask, other=0.0, eviction_policy="evict_last"
        )  # Want T to stay resident in L2 cache
    else:
        T_C = tl.load(
            h_ptr + group_num * FILTER_LEN + t_idx,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )

    return T_C
