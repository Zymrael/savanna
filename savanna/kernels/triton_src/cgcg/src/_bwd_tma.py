import logging
from typing import Any, Union

import torch
import triton
import triton.language as tl

from savanna.kernels.triton_src.cgcg.src.bwd_kernels import _get_T_store_idx, _get_Tc_store_idx
from savanna.kernels.triton_src.cgcg.src.fwd_kernels import load_correction_toeplitz, load_toeplitz
from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    # DEVICE_PROPS,
    # BwdKernelResult,
    create_2d_tma_descriptor,
    get_program_order,
    torch_dtype_to_triton,
)

logger = logging.getLogger(__name__)


# TODO: Threshold for using TMA for filters: g * hl * element_size >= 128
@triton.jit
def _two_pass_bwd_grouped_tma_kernel(
    # TMA descriptors
    # Inputs
    dy_desc,
    x_desc,
    B_desc,
    C_desc,
    h_ptr,
    # Intermediate activations
    y2_desc,
    T_desc,
    T_hat_desc,
    # Outputs
    dx_desc,
    dB_desc,
    dC_desc,
    dhdT_ptr,
    dhdTc_ptr,
    # Strides
    input_batch_stride,
    input_row_stride,
    input_col_stride,
    dhdT_batch_stride,
    dhdT_chunk_stride,
    dhdT_block_stride,
    dhdT_row_stride,
    dhdT_col_stride,
    # Shapes
    bs,
    seqlen,
    g,
    dg,
    # Compile-time constants
    FILTER_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DTYPE: tl.constexpr,
    LOAD_TOEPLITZ: tl.constexpr,
    SINGLE_GROUP: tl.constexpr,
    NUM_PIPELINE_STAGES: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    FLUSH: tl.constexpr = False,  # Flush TMA cache
    # kwargs for tl.dot
    input_precision: tl.constexpr = "ieee",  # "ieee", "tf32", "tf32x3"  --> only for debugging, since dtype < fp32
    max_num_imprecise_acc: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32,
    # Common triton kernel params
    # num_stages: tl.constexpr = 2,
    # num_warps: tl.constexpr = 4,
    DEBUG: tl.constexpr = False,
):
    # tl.static_print("DTYPE", DTYPE,)
    # tl.static_print("DTYPE.value", DTYPE.value)

    if DEBUG:
        if tl.program_id(0) == 0:
            tl.static_print(
                "TWO_PASS CONSTEXPRS:\n",
                "FILTER_LEN:",
                FILTER_LEN,
                "CHUNK_SIZE:",
                CHUNK_SIZE,
                "BLOCK_D:",
                BLOCK_D,
                "SINGLE_GROUP:",
                SINGLE_GROUP,
                "THREADBLOCK_SWIZZLE:",
                THREADBLOCK_SWIZZLE,
            )
            # tl.static_print("DTYPE", DTYPE,)
            # tl.static_print("DTYPE.value", DTYPE.value)

    d = g * dg
    num_programs = tl.num_programs(0)
    # TMA offsets
    pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(seqlen, CHUNK_SIZE)
    num_tiles_k = tl.cdiv(d, BLOCK_D)
    num_tiles_batch = num_tiles_m * num_tiles_k
    total_tiles = bs * num_tiles_batch
    chunks_per_batch = num_tiles_m
    tiles_per_filter_group = dg // BLOCK_D
    chunks_per_seq = tl.cdiv(seqlen, CHUNK_SIZE)

    for tile_id in tl.range(pid, total_tiles, num_programs, num_stages=NUM_PIPELINE_STAGES):
        pid_batch, pid_chunk, pid_d, pid_filter_group = get_program_order(
            tile_id,
            num_tiles_batch,
            num_tiles_k,
            chunks_per_batch,
            tiles_per_filter_group,
            THREADBLOCK_SWIZZLE,
        )

        batch_offset = pid_batch * num_tiles_m * CHUNK_SIZE
        chunk_offset = pid_chunk * CHUNK_SIZE
        offset_m = batch_offset + chunk_offset
        offset_k = pid_d * BLOCK_D

        if False:
            if pid == 0:
                tl.device_print("num_pid_m", num_tiles_m)
                tl.device_print("num_pid_k", num_tiles_k)
                tl.device_print("num_pid_batch", num_tiles_batch)
                tl.device_print("total_tiles", total_tiles)

            tl.device_print("pid", pid)
            tl.device_print("pid_batch", pid_batch)
            tl.device_print("pid_chunk", pid_chunk)
            tl.device_print("pid_d", pid_d)
            tl.device_print("pid_filter_group", pid_filter_group)
            tl.device_print("offset_m", offset_m)
            tl.device_print("offset_k", offset_k)

        dy = tl._experimental_descriptor_load(
            dy_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
        )

        x = tl._experimental_descriptor_load(x_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value)
        B = tl._experimental_descriptor_load(B_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value)
        C = tl._experimental_descriptor_load(C_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value)
        y2 = tl._experimental_descriptor_load(
            y2_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
        )

        # Start backprop
        dC = dy * y2
        tl._experimental_descriptor_store(dC_desc, dC, [offset_m, offset_k])

        # # Backprop through C
        dy = dy * C

        if LOAD_TOEPLITZ:
            T_group_stride = CHUNK_SIZE
            T_group_offset = pid_filter_group * T_group_stride

            T = tl._experimental_descriptor_load(
                T_desc, [T_group_offset, 0], [CHUNK_SIZE, CHUNK_SIZE], DTYPE.value
            )
        else:
            T = load_toeplitz(
                h_ptr,
                FILTER_LEN,
                CHUNK_SIZE,
                SINGLE_GROUP=SINGLE_GROUP,
                group_num=pid_filter_group,
            )

        T = tl.trans(T)
        dy1 = tl.dot(
            T,
            dy,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=dy.dtype,
        )
        dx = dy1 * B
        dB = dy1 * x

        Bx = tl.trans(B * x)
        dT = tl.dot(
            dy,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=dy.dtype,
        )

        # Correction term
        # In backwards, we roll in the opposite direction
        # Hence, the last chunk in the sequence does not need correction
        is_last_chunk = pid_chunk == chunks_per_seq - 1
        is_first_chunk = pid_chunk == 0

        if not is_last_chunk:
            offset_m_lead = (
                offset_m + CHUNK_SIZE
            )  # offset_m = batch_offset + chunk_offset = batch_offset + (pid_chunk - 1) * CHUNK_SIZE
            dy_lead = tl._experimental_descriptor_load(
                dy_desc, [offset_m_lead, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
            )
            C_lead = tl._experimental_descriptor_load(
                C_desc, [offset_m_lead, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
            )
            dy_lead *= C_lead

            if LOAD_TOEPLITZ:
                T_c = tl._experimental_descriptor_load(
                    T_hat_desc,
                    [T_group_offset, 0],
                    [CHUNK_SIZE, CHUNK_SIZE],
                    DTYPE.value,
                )
            else:
                T_c = load_correction_toeplitz(
                    h_ptr,
                    FILTER_LEN,
                    CHUNK_SIZE,
                    SINGLE_GROUP=SINGLE_GROUP,
                    group_num=pid_filter_group,
                )
            T_c = tl.trans(T_c)

            dcorrection = tl.dot(
                T_c,
                dy_lead,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=dy.dtype,
            )
            dcorrection_dx = dcorrection * B
            dcorrection_dB = dcorrection * x
            dx += dcorrection_dx
            dB += dcorrection_dB

        tl._experimental_descriptor_store(dx_desc, dx, [offset_m, offset_k])
        tl._experimental_descriptor_store(dB_desc, dB, [offset_m, offset_k])

        # Store dhdT
        dhdT_idx, dhdT_mask = _get_T_store_idx(
            CHUNK_SIZE, FILTER_LEN, row_stride=dhdT_row_stride, col_stride=1
        )

        dhdT_offsets = (
            pid_batch * dhdT_batch_stride
            + pid_chunk * dhdT_chunk_stride
            + pid_d * dhdT_block_stride
            + dhdT_idx
        )
        tl.store(dhdT_ptr + dhdT_offsets, dT, mask=dhdT_mask)

        # num chunks per seq * num blocks per d
        # dT_batch_stride = num_tiles_m * num_tiles_k * FILTER_LEN
        # dT_chunk_stride = num_tiles_k * FILTER_LEN
        # dT_block_stride = FILTER_LEN
        # dT_offset_m = pid_batch * dT_batch_stride + pid_chunk * dT_chunk_stride + pid_d * dT_block_stride
        # tl._experimental_descriptor_store(dhdT_desc, dT, [dT_offset_m, 0])

        if not is_first_chunk:
            offset_m_lag = offset_m - CHUNK_SIZE
            B_lag = tl._experimental_descriptor_load(
                B_desc, [offset_m_lag, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
            )
            x_lag = tl._experimental_descriptor_load(
                x_desc, [offset_m_lag, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
            )
            Bx_lag = B_lag * x_lag
            Bx_lag = tl.trans(Bx_lag)
            dTc = tl.dot(
                dy,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=dy.dtype,
            )

            dhdTc_idx, dhdTc_mask = _get_Tc_store_idx(
                CHUNK_SIZE, FILTER_LEN, row_stride=dhdT_row_stride, col_stride=1
            )
            dhdTc_offsets = (
                pid_batch * dhdT_batch_stride
                + pid_chunk * dhdT_chunk_stride
                + pid_d * dhdT_block_stride
                + dhdTc_idx
            )
            tl.store(dhdTc_ptr + dhdTc_offsets, dTc, mask=dhdTc_mask)


def two_pass_bwd_grouped_tma(
    dy: torch.Tensor,
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    y2: torch.Tensor,
    T: torch.Tensor = None,
    T_hat: torch.Tensor = None,
    version: str = "v1",
    schedule: str = "default",
    autotune: bool = False,
    CHUNK_SIZE: int = None,
    BLOCK_D: int = None,
    NUM_PIPELINE_STAGES: int = 0,  # for tl.range
    THREADBLOCK_SWIZZLE: str = "row",
    num_warps: int = None,
    # TODO: Make sure to set match these defaults to those in CUDAOptions
    num_stages: int = 3,  # for tl.dot, should default to 3
    num_ctas: int = 1,
    maxnreg: int = None,
    warmup: bool = False,
    return_kernel: bool = False,
) -> Union[torch.tensor, tuple[triton.compiler.CompiledKernel, tuple[Any], tuple[Any]]]:
    """
    Chunked two-pass backwards kernel with grouped filters with TMA enabled, only for sm90+

    See `cgcg.triton.bwd_kernels.two_pass_bwd_grouped` for documentation on args and return values.
    """

    bs, seqlen, g, dg = dy.shape
    filter_shape = h.shape
    hg, _in_channel_div_group, filter_len = filter_shape

    if autotune:
        raise NotImplementedError("Autotuning not implemented for bwd")
    else:
        assert all(
            [
                CHUNK_SIZE,
                BLOCK_D,
                num_warps,
                num_stages is not None,
                NUM_PIPELINE_STAGES is not None,
            ]
        ), "Must specify all of CHUNK_SIZE, BLOCK_D, NUM_PIPELINE_STAGES, num_warps, num_stages, "
        if version == "v1":
            kernel: triton.runtime.JITFunction = _two_pass_bwd_grouped_tma_kernel
        elif version == "v2":
            raise NotImplementedError("v2 not implemented yet")
        else:
            raise ValueError(f"version {version} not implemented")

    if CHUNK_SIZE < filter_len:
        raise ValueError("CHUNK_SIZE must be >= filter_len")

    # basic shape checks
    assert dg >= 16, "dg must be >= 8 to use tensor-cores"
    assert x.shape == dy.shape == B.shape == C.shape == y2.shape
    assert hg == g
    assert _in_channel_div_group == 1

    # hidden_dim
    d = g * dg

    x = x.reshape(bs, seqlen, d)
    B = B.reshape_as(x)
    C = C.reshape_as(x)
    dy = dy.reshape_as(x)

    # Intermediates from forward pass
    y2 = y2.reshape_as(x)
    batch_stride, row_stride, col_stride = dy.stride()

    if T is not None:
        assert T_hat is not None
        assert T.shape == T_hat.shape == torch.Size([g, CHUNK_SIZE, CHUNK_SIZE])
        assert T.is_contiguous()
        assert T_hat.is_contiguous()
        # Kernel constexpr
        LOAD_TOEPLITZ = True
        # Create 2D TMA descriptors for loading T, T_hat since 1-D TMA descriptor can't be used when
        # block size must less than 256 elements, see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
        # Indexing logic will be only along M dimension
        # Stride by CHUNK_SIZE along M to advance each block, where each block maps to a filter group
        T_M = g * CHUNK_SIZE
        T_K = CHUNK_SIZE
        # logger.debug("Creating 2D TMA descriptors for T, T_hat")
        desc_T = create_2d_tma_descriptor(T.data_ptr(), T_M, T_K, CHUNK_SIZE, CHUNK_SIZE, T.element_size())
        desc_T_hat = create_2d_tma_descriptor(
            T_hat.data_ptr(), T_M, T_K, CHUNK_SIZE, CHUNK_SIZE, T_hat.element_size()
        )
    else:
        LOAD_TOEPLITZ = False
        desc_T = None
        desc_T_hat = None

    # triton kernel pre-conditions
    assert dy.is_contiguous()
    assert x.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert y2.is_contiguous()

    # Reshape h to a 2-D tensor
    # TODO: remove?
    h = h.reshape(g, filter_len)
    assert h.is_contiguous()

    # use_autotuner = not any([CHUNK_SIZE, BLOCK_D, num_warps, NUM_PIPELINE_STAGES])
    assert not (
        autotune and warmup
    ), "autotune and warmup are not supported, use return_kernel=True to get the kernel after autotuning"

    if schedule == "default":

        def _1d_grid(META):
            row_tiles = triton.cdiv(seqlen, META["CHUNK_SIZE"])
            col_tiles = triton.cdiv(d, META["BLOCK_D"])
            total_tiles = bs * row_tiles * col_tiles
            return (total_tiles,)

        # logger.info("Setting NUM_PIPELINE_STAGES = 0 since schedule is `default`")
        NUM_PIPELINE_STAGES = 0
        grid = _1d_grid

    elif schedule == "persistent":
        raise NotImplementedError("Skip persistent for now")
        # grid = lambda META: (
        #     min(
        #         DEVICE_PROPS.NUM_SM,
        #         triton.cdiv(seqlen, META["CHUNK_SIZE"])
        #         * triton.cdiv(d, META["BLOCK_D"])
        #         * bs,
        #     ),
        # )
    else:
        raise ValueError(f"schedule {schedule} not implemented")

    # Create TMA descriptors
    M, K = bs * seqlen, d

    # Load
    # logger.debug("Creating 2D TMA descriptors for dy, x, B, C, y2")

    desc_dy = create_2d_tma_descriptor(dy.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, dy.element_size())
    desc_x = create_2d_tma_descriptor(x.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, x.element_size())
    desc_B = create_2d_tma_descriptor(B.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, B.element_size())
    desc_C = create_2d_tma_descriptor(C.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, C.element_size())
    desc_y2 = create_2d_tma_descriptor(y2.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, y2.element_size())
    # 1D TMA requirement: elementSize * numel() >= 128
    # desc_h = create_1d_tma_descriptor(h.data_ptr(), g * hl, hl, h.element_size())

    # Store
    dx = torch.zeros_like(x)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    desc_dx = create_2d_tma_descriptor(dx.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, dx.element_size())
    desc_dB = create_2d_tma_descriptor(dB.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, dB.element_size())
    desc_dC = create_2d_tma_descriptor(dC.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, dC.element_size())

    num_chunks = triton.cdiv(seqlen, CHUNK_SIZE)
    num_blocks = triton.cdiv(d, BLOCK_D)

    if version == "v1":
        dhdT = torch.zeros(
            bs,
            num_chunks,
            num_blocks,
            filter_len,
            CHUNK_SIZE,
            device=h.device,
            dtype=h.dtype,
        )
        dhdT_hat = torch.zeros_like(dhdT)
        (
            dhdT_batch_stride,
            dhdT_chunk_stride,
            dhdT_block_stride,
            dhdT_row_stride,
            dhdT_col_stride,
        ) = dhdT.stride()

        # NOTE: can't use TMA since we are slicing segments of dhdT/dhdT_hat during store
        # 1D TMA descriptor does not work once filter_len > 6, hence we use 2D TMA
        # M is equal to bs * num_chunks * num_blocks * filter_len
        # K is equal to CHUNK_SIZE
        # block sizes are filter_len x CHUNK_SIZE
        # This means we only to need to advance along m dimension when storing
        # Calculate bs, chunk, and block ids by pid
        # Advance by filter_len along M
        # offset along k dimension should be 0
        # dh_M = bs * num_chunks * num_blocks * filter_len
        # dh_K = CHUNK_SIZE
        # #
        # desc_dhdT = create_2d_tma_descriptor(
        #     dhdT.data_ptr(), dh_M, dh_K, filter_len, CHUNK_SIZE, dhdT.element_size()
        # )
        # desc_dhdT_hat = create_2d_tma_descriptor(
        #     dhdT_hat.data_ptr(),
        #     dh_M,
        #     dh_K,
        #     filter_len,
        #     CHUNK_SIZE,
        #     dhdT_hat.element_size(),
        # )

        kernel_args = (
            # Inputs
            desc_dy,
            desc_x,
            desc_B,
            desc_C,
            h,
            # Intermediate activations
            desc_y2,
            desc_T,
            desc_T_hat,
            # Outputs
            desc_dx,
            desc_dB,
            desc_dC,
            dhdT,
            dhdT_hat,
            # Strides
            batch_stride,
            row_stride,
            col_stride,
            dhdT_batch_stride,
            dhdT_chunk_stride,
            dhdT_block_stride,
            dhdT_row_stride,
            dhdT_col_stride,
            # Shapes
            bs,
            seqlen,
            g,
            dg,
        )

    else:
        dh_buffers = torch.zeros(bs, num_chunks, num_blocks, filter_len, device=x.device, dtype=h.dtype)
        dh_batch_stride, dh_chunk_stride, dh_block_stride, _ = dh_buffers.stride()

        kernel_args = (
            dy,
            x,
            B,
            C,
            h,
            # Intermediate activations
            y2,
            T,
            T_hat,
            # Outputs
            dx,
            dB,
            dC,
            dh_buffers,
            # Strides
            batch_stride,
            row_stride,
            col_stride,
            dh_batch_stride,
            dh_chunk_stride,
            dh_block_stride,
            # Shapes
            bs,
            seqlen,
            g,
            dg,
        )

    # TODO: check dtypes all the same
    kernel_constexprs = {
        "FILTER_LEN": filter_len,
        "SINGLE_GROUP": g == 1,
        "LOAD_TOEPLITZ": LOAD_TOEPLITZ,
        "DTYPE": torch_dtype_to_triton(x.dtype),
    }

    if not autotune:
        kernel_constexprs.update(
            {
                "CHUNK_SIZE": CHUNK_SIZE,
                "BLOCK_D": BLOCK_D,
                "THREADBLOCK_SWIZZLE": THREADBLOCK_SWIZZLE,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "num_ctas": num_ctas,
                "NUM_PIPELINE_STAGES": NUM_PIPELINE_STAGES,
            }
        )

    # Can actually run this with fake tensors (no need for actual kernel tensor args)
    if warmup:
        compiled_kernel: triton.compiler.CompiledKernel = kernel.warmup(
            *kernel_args, **kernel_constexprs, grid=(1,)
        )
        return compiled_kernel, kernel_args, kernel_constexprs
    else:
        # results = []
        # logger.info(
        #     f"Running backward kernel {version} with {schedule=} {kernel_constexprs=}"
        # )

        # Run the kernel
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](*kernel_args, **kernel_constexprs)

        dx = dx.reshape(bs, seqlen, g, dg)
        dB = dB.reshape(bs, seqlen, g, dg)
        dC = dC.reshape(bs, seqlen, g, dg)

        num_blocks_per_filter_group = dg // BLOCK_D

        # Run second final reduction kernel for dh
        # TODO: either `torch.compile`` or write custom triton kernel for this
        if version == "v1":
            dhdT = dhdT.reshape(bs, num_chunks, g, num_blocks_per_filter_group, filter_len, CHUNK_SIZE)
            dhdT_hat = dhdT_hat.reshape_as(dhdT)
            dhdT = dhdT.sum([0, 1, 3, 5]).reshape(*filter_shape)
            dhdTc = dhdT_hat.sum([0, 1, 3, 5]).reshape_as(dhdT)
            dh = dhdT + dhdTc

        elif version == "v2":
            dh_buffers = dh_buffers.reshape(bs, num_chunks, g, num_blocks_per_filter_group, filter_len)
            dh = dh_buffers.sum([0, 1, 3]).reshape(*filter_shape)

        if return_kernel:
            return dx, dB, dC, dh, compiled_kernel
        else:
            return dx, dB, dC, dh


# if __name__ == "__main__":
#     from bwd_kernels import two_pass_bwd_grouped

#     from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected, gcg_two_pass_chunked_fwd_corrected
#     from savanna.kernels.triton_src.cgcg.utils import correction_toeplitz, toeplitz

#     bs, seqlen, d = (2, 128, 128)
#     g = 2
#     hl = 4
#     CHUNK_SIZE = 32
#     BLOCK_D = 32
#     dtype = torch.float32
#     LOAD_TOEPLITZ = False
#     schedule = "default"

#     dg = d // g
#     num_warps = 4  # if filter_size < 128 else 2
#     num_stages = 2  # 1 if filter_size > 6 else 2
#     swizzle = "row"
#     autotune = False

#     def setup_inputs(bs, seqlen, dg, g, filter_size, dtype, requires_grad=True):
#         device = "cuda"
#         x = torch.randn(
#             bs, seqlen, g, dg, device=device, dtype=dtype, requires_grad=requires_grad
#         )
#         B = torch.randn(
#             bs, seqlen, g, dg, device=device, dtype=dtype, requires_grad=requires_grad
#         )
#         C = torch.randn(
#             bs, seqlen, g, dg, device=device, dtype=dtype, requires_grad=requires_grad
#         )
#         h = torch.randn(
#             g * filter_size, device=device, dtype=dtype, requires_grad=requires_grad
#         ).reshape(g, 1, filter_size)
#         return x, B, C, h

#     x, B, C, h = setup_inputs(bs, seqlen, dg, g, hl, dtype)
#     # Ref grad
#     x_ref = x.detach().clone().requires_grad_()
#     B_ref = B.detach().clone().requires_grad_()
#     C_ref = C.detach().clone().requires_grad_()
#     h_ref = h.detach().clone().requires_grad_()

#     # We need y2 = T_local @ Bx + T_correction @ Bx_lag to calculate dC
#     # We can't use the chunked ref for calculating dh since h becomes detached when constructing T_local and T_c
#     _, _, _, y2, _ = gcg_two_pass_chunked_fwd_corrected(
#         x_ref.detach().clone(),
#         B_ref.detach().clone(),
#         C_ref.detach().clone(),
#         h_ref.detach().clone(),
#         gl=CHUNK_SIZE,
#         return_intermediates=True,
#     )

#     if LOAD_TOEPLITZ:
#         h_ = h.flip(-1)[:, 0]
#         T = toeplitz(h_, CHUNK_SIZE)
#         T_hat = correction_toeplitz(h_, CHUNK_SIZE)
#     else:
#         T = None
#         T_hat = None

#     y_ref = gcg_fwd_ref_corrected(x_ref, B_ref, C_ref, h_ref)

#     # Backprop
#     dy = 0.1 * torch.randn_like(y_ref)
#     y_ref.backward(dy)
#     dx_ref = x_ref.grad
#     dB_ref = B_ref.grad
#     dC_ref = C_ref.grad
#     dh_ref = h_ref.grad

#     kernel_config = {
#         "CHUNK_SIZE": CHUNK_SIZE,
#         "BLOCK_D": BLOCK_D,
#         "num_warps": num_warps,
#         "NUM_PIPELINE_STAGES": 0 if schedule == "default" else 1,
#         "num_stages": num_stages,
#         "THREADBLOCK_SWIZZLE": swizzle,
#     }

#     dx_ref, dB_ref, dC_ref, dT_ref, dTc_ref, dh_ref_bwd = two_pass_bwd_grouped(
#         dy,
#         x_ref,
#         B_ref,
#         C_ref,
#         h_ref,
#         y2,
#         T=T,
#         T_hat=T_hat,
#         schedule=schedule,
#         autotune=autotune,
#         **kernel_config,
#     )
#     # Test grad
#     dx, dB, dC, dT, dTc, dh = two_pass_bwd_grouped_tma(
#         dy,
#         x,
#         B,
#         C,
#         h,
#         y2,
#         T=T,
#         T_hat=T_hat,
#         schedule=schedule,
#         autotune=autotune,
#         **kernel_config,
#     )
#     # print(f"dx_ref: {dx_ref}")
#     # print(f"dC_ref: {dC_ref}")
#     # print(f"dC: {dC}")
#     x_diff = (dx - dx_ref).abs().max()
#     print(f"x_diff: {x_diff}")
#     B_dff = (dB - dB_ref).abs().max()
#     print(f"B_diff: {B_dff}")
#     print(f"dC diff: {(dC - dC_ref).abs().max()}")
#     print(f"dT diff: {(dT - dT_ref).abs().max()}")
#     print(f"dTc diff: {(dTc - dTc_ref).abs().max()}")
#     print(f"dh diff: {(dh_ref_bwd - dh).abs().max()}")
#     print(f"dh_bwd diff: {(dh_ref - dh_ref_bwd).abs().max()}")
#     print(f"dh_bwd_tma diff: {(dh_ref - dh).abs().max()}")

# # B_diff = (dB - B).abs().max()
# # C_diff = (dC - C).abs().max()
# # print(f"x_diff: {x_diff}, B_diff: {B_diff}, C_diff: {C_diff}")
