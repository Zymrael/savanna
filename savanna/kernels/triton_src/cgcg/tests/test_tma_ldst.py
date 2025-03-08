import pytest
import torch
import triton
import triton.language as tl

from savanna.kernels.triton_src.cgcg.src.kernel_utils import create_2d_tma_descriptor


@triton.jit
def matmul_kernel_tma(
    x_desc_ptr,
    out_desc_ptr,
    bs,
    seqlen,
    d,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FLUSH: tl.constexpr = False,
    DEBUG: tl.constexpr = False,
):
    # TODO(embg) remove TMA fence after __grid_constant__ lands
    if FLUSH:
        tl.inline_asm_elementwise(
            "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
            "=r, l",
            [x_desc_ptr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.inline_asm_elementwise(
            "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
            "=r, l",
            [out_desc_ptr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(seqlen, BLOCK_SIZE_M)
    num_tiles_k = tl.cdiv(d, BLOCK_SIZE_K)
    num_tiles_batch = num_tiles_m * num_tiles_k
    chunks_per_batch = num_tiles_m

    pid_batch = pid // num_tiles_batch
    pid_chunk = (pid // num_tiles_k) % chunks_per_batch
    pid_d = pid % num_tiles_k

    batch_offset = pid_batch * num_tiles_m * BLOCK_SIZE_M
    chunk_offset = pid_chunk * BLOCK_SIZE_M
    m_offset = batch_offset + chunk_offset
    k_offset = pid_d * BLOCK_SIZE_K

    if DEBUG:
        if pid == 0:
            tl.device_print("num_pid_m", num_tiles_m)
            tl.device_print("num_pid_k", num_tiles_k)
            tl.device_print("num_pid_batch", num_tiles_batch)
            tl.device_print("total_tiles", num_tiles_batch * bs)

        tl.device_print("pid", pid)
        tl.device_print("pid_batch", pid_batch)
        tl.device_print("pid_chunk", pid_chunk)
        tl.device_print("pid_d", pid_d)
        tl.device_print("offs_am", m_offset)
        tl.device_print("offs_ak", k_offset)

    x = tl._experimental_descriptor_load(
        x_desc_ptr, [m_offset, k_offset], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16
    )
    tl._experimental_descriptor_store(out_desc_ptr, x, [m_offset, k_offset])


@pytest.mark.parametrize(
    "bs, seqlen, d, BLOCK_SIZE_M, BLOCK_SIZE_K, dtype, FLUSH",
    [
        (2, 1024, 768, 32, 32, torch.float16, False),
        (2, 1024, 768, 32, 32, torch.float16, True),
    ],
)
def test_tma_ldst(bs, seqlen, d, BLOCK_SIZE_M, BLOCK_SIZE_K, dtype, FLUSH):
    x = torch.randn(bs * seqlen * d).reshape(bs, seqlen, d).to(dtype).cuda()
    out = torch.empty_like(x)

    M, K = bs * seqlen, d

    desc_x = create_2d_tma_descriptor(x.data_ptr(), M, K, BLOCK_SIZE_M, BLOCK_SIZE_K, x.element_size())
    desc_out = create_2d_tma_descriptor(out.data_ptr(), M, K, BLOCK_SIZE_M, BLOCK_SIZE_K, out.element_size())

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_K), 1, 1)
    matmul_kernel_tma[grid](
        desc_x,
        desc_out,
        bs,
        seqlen,
        d,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=1,
    )
    assert torch.allclose(x, out)
