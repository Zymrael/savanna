from typing import Any, Union

import torch
import triton
import triton.language as tl

from savanna.kernels.triton_src.cgcg.src.kernel_utils import (
    # DEVICE_PROPS,
    # AutotunedResult,
    # FwdKernelResult,
    create_2d_tma_descriptor,
    get_program_order,
    torch_dtype_to_triton,
)
from savanna.kernels.triton_src.cgcg.src.toeplitz_kernels import load_correction_toeplitz, load_toeplitz


@triton.jit
def _two_pass_fwd_grouped_tma_kernel(
    # Pointers
    x_desc,
    B_desc,
    C_desc,
    h_ptr,
    y_desc,
    T_ptr,
    T_hat_ptr,
    y2_desc,
    # Strides
    batch_stride,
    row_stride,
    col_stride,
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
    NUM_PIPELINE_STAGES: tl.constexpr,
    THREADBLOCK_SWIZZLE: tl.constexpr,
    SINGLE_GROUP: tl.constexpr,
    RETURN_TOEPLITZ: tl.constexpr = False,
    RETURN_Y2: tl.constexpr = False,
    FLUSH: tl.constexpr = False,
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
    tl.static_print("USE_TMA")
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

        x = tl._experimental_descriptor_load(x_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value)
        B = tl._experimental_descriptor_load(B_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value)

        Bx = B * x
        T = load_toeplitz(
            h_ptr,
            FILTER_LEN,
            CHUNK_SIZE,
            SINGLE_GROUP=SINGLE_GROUP,
            group_num=pid_filter_group,
        )
        y = tl.dot(
            T,
            Bx,
            input_precision=input_precision,
            max_num_imprecise_acc=max_num_imprecise_acc,
            out_dtype=DTYPE.value,
        )

        if RETURN_TOEPLITZ:
            blocks_per_filter_group = dg // BLOCK_D
            is_first_pid_in_group = (pid_d % blocks_per_filter_group) == 0
            t_group_stride = CHUNK_SIZE * CHUNK_SIZE
            t_offsets = (
                pid_filter_group * t_group_stride
                + tl.arange(0, CHUNK_SIZE)[:, None] * CHUNK_SIZE
                + tl.arange(0, CHUNK_SIZE)[None, :]
            )
            # Only need to store once per batch per seq per filter group
            if pid_batch == 0 and pid_chunk == 0:
                # Only first program for each filter group needs to store
                if is_first_pid_in_group:
                    tl.store(T_ptr + t_offsets, T)

        if pid_chunk > 0:
            T_c = load_correction_toeplitz(
                h_ptr,
                FILTER_LEN,
                CHUNK_SIZE,
                SINGLE_GROUP=SINGLE_GROUP,
                group_num=pid_filter_group,
            )

            offset_m_lag = (
                offset_m - CHUNK_SIZE
            )  # offset_m = batch_offset + chunk_offset = batch_offset + (pid_chunk - 1) * CHUNK_SIZE
            x_lag = tl._experimental_descriptor_load(
                x_desc, [offset_m_lag, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
            )
            B_lag = tl._experimental_descriptor_load(
                B_desc, [offset_m_lag, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value
            )
            Bx_lag = B_lag * x_lag

            correction_term = tl.dot(
                T_c,
                Bx_lag,
                input_precision=input_precision,
                max_num_imprecise_acc=max_num_imprecise_acc,
                out_dtype=DTYPE.value,
            )

            y += correction_term

            if RETURN_TOEPLITZ:
                # First chunk doesn't calculate correction toeplitz
                # pid_batch = 0, pid_chunk = 1 is the first batch / chunk that calculates correction term
                if pid_batch == 0 and pid_chunk == 1:
                    if is_first_pid_in_group:
                        tl.store(T_hat_ptr + t_offsets, T_c)

        if RETURN_Y2:
            tl._experimental_descriptor_store(y2_desc, y, [offset_m, offset_k])

        C = tl._experimental_descriptor_load(C_desc, [offset_m, offset_k], [CHUNK_SIZE, BLOCK_D], DTYPE.value)
        y *= C

        tl._experimental_descriptor_store(y_desc, y, [offset_m, offset_k])


# TODO: precompile kernel for persistent schedule to maximize occupancy
def two_pass_fwd_grouped_tma(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    y: torch.Tensor = None,
    return_y2: bool = False,  # y2 = T_local @ B*x + T_hat @ B_lag*x_lag
    return_toeplitz: bool = False,
    verbose: bool = False,
    version: str = "v1",
    schedule: str = "default",
    autotune: bool = False,
    CHUNK_SIZE: int = None,
    BLOCK_D: int = None,
    NUM_PIPELINE_STAGES: int = 0,  # for tl.range
    THREADBLOCK_SWIZZLE: str = "row",
    CHUNK_TILES_PER_PROGRAM: int = 1,
    num_warps: int = None,
    # TODO: Make sure to set match these defaults to those in CUDAOptions
    num_stages: int = 3,  # for tl.dot, should default to 3
    num_ctas: int = 1,
    maxnreg: int = None,
    warmup: bool = False,
    return_kernel: bool = False,
    return_autotune_result: bool = False,
    DEBUG: bool = False,
) -> Union[torch.tensor, tuple[triton.compiler.CompiledKernel, tuple[Any], tuple[Any]]]:
    """
    Chunked two-pass forward kernel with grouped filters with TMA enabled

    See `cgcg.triton.fwd_kernels.two_pass_fwd_grouped` for documentation on args and return values.
    """
    bs, seqlen, g, dg = x.shape

    # basic shape checks
    assert dg >= 16, "dg must be >= 8 to use tensor-cores"
    assert x.shape == B.shape == C.shape
    hg, _in_channel_div_group, filter_len = h.shape
    assert hg == g
    assert _in_channel_div_group == 1

    # hidden_dim
    d = g * dg

    x = x.reshape(bs, seqlen, d)
    B = B.reshape_as(x)
    C = C.reshape_as(x)
    batch_stride, row_stride, col_stride = x.stride()

    # triton kernel pre-condition
    assert x.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()

    assert (
        x.data_ptr() % 16 == 0 and B.data_ptr() % 16 == 0 and C.data_ptr() % 16 == 0
    ), "Pointers must be 16-byte aligned to use TMA"

    if not h.data_ptr() % 16 == 0:
        print("WARNING: h must be 16-byte aligned to use TMA, not using TMA for h")

    M, K = bs * seqlen, d

    desc_x = create_2d_tma_descriptor(x.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, x.element_size())

    desc_B = create_2d_tma_descriptor(B.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, B.element_size())
    desc_C = create_2d_tma_descriptor(C.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, C.element_size())

    if y is None:
        y = torch.zeros_like(x)

    desc_y = create_2d_tma_descriptor(y.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, y.element_size())
    # Reshape h to a 2-D tensor
    # TODO: remove?
    h = h.reshape(g, filter_len)
    assert h.is_contiguous()

    assert not (
        autotune and warmup
    ), "autotune and warmup are not supported, use return_kernel=True to get the kernel after autotuning"

    if autotune:
        raise NotImplementedError("autotuning not implemented for TMA")
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
        kernel: triton.runtime.JITFunction = _two_pass_fwd_grouped_tma_kernel

    if BLOCK_D is not None:
        assert dg % BLOCK_D == 0, "dg must be multiple of BLOCK_D"

    if schedule == "default":

        def _1d_grid(META):
            row_tiles = triton.cdiv(seqlen, META["CHUNK_SIZE"])
            col_tiles = triton.cdiv(d, META["BLOCK_D"])
            total_tiles = bs * row_tiles * col_tiles
            return (total_tiles,)

        NUM_PIPELINE_STAGES = 0
        grid = _1d_grid
    # elif schedule == "persistent":

    #     grid = lambda META: (
    #         min(
    #             DEVICE_PROPS.NUM_SM,
    #             triton.cdiv(seqlen, META["CHUNK_SIZE"])
    #             * triton.cdiv(d, META["BLOCK_D"])
    #             * bs,
    #         ),
    #     )
    else:
        raise ValueError(f"schedule {schedule} not implemented")

    # For backwards
    if return_y2:
        y2 = torch.empty_like(x)
        desc_y2 = create_2d_tma_descriptor(y2.data_ptr(), M, K, CHUNK_SIZE, BLOCK_D, y2.element_size())
    else:
        y2 = None
        desc_y2 = None

    if return_toeplitz:
        assert CHUNK_SIZE is not None, "CHUNK_SIZE must be specified for return_toeplitz"
        # NOTE: Need to initialize T_hat as zeros, since not all chunks need correction term
        T = torch.zeros(g, CHUNK_SIZE, CHUNK_SIZE, device=x.device, dtype=x.dtype)
        T_hat = torch.zeros_like(T)
    else:
        T = None
        T_hat = None

    DTYPE = torch_dtype_to_triton(x.dtype)
    if verbose:
        #  print(f"{x.shape=}, {B.shape=}, {C.shape=}, {h.shape=}, {y.shape=}")
        print(f"{bs=} {seqlen=} {g=} {dg=} {filter_len=}")
        print(f"{CHUNK_SIZE=}, {BLOCK_D=}, {num_warps=}, {NUM_PIPELINE_STAGES=} {DTYPE=}")

    kernel_args = (
        desc_x,
        desc_B,
        desc_C,
        h,
        desc_y,
        T,
        T_hat,
        desc_y2,
        batch_stride,
        row_stride,
        col_stride,
        bs,
        seqlen,
        g,
        dg,
    )

    kernel_constexprs = {
        "FILTER_LEN": filter_len,
        "SINGLE_GROUP": g == 1,
        "RETURN_TOEPLITZ": return_toeplitz,
        "RETURN_Y2": return_y2,
        "DTYPE": DTYPE,
        "DEBUG": DEBUG,
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

    if warmup:
        compiled_kernel: triton.compiler.CompiledKernel = kernel.warmup(
            *kernel_args, **kernel_constexprs, grid=(1,)
        )
        return compiled_kernel, kernel_args, kernel_constexprs
    else:
        # Run the kernel
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](*kernel_args, **kernel_constexprs)

        y = y.reshape(bs, seqlen, g, dg)
        if y2 is not None:
            y2 = y2.reshape(bs, seqlen, g, dg)

        if not return_kernel:
            return y, T, T_hat, y2
        else:
            return y, T, T_hat, y2, compiled_kernel


# if __name__ == "__main__":
#     from fwd_kernels import two_pass_fwd_grouped

#     from savanna.kernels.triton_src.cgcg.ref_fwd import gcg_fwd_ref_corrected, gcg_two_pass_chunked_fwd_corrected
#     from savanna.kernels.triton_src.cgcg.triton.kernel_utils import get_kernel_occupancy
#     from savanna.kernels.triton_src.cgcg.utils import correction_toeplitz, toeplitz
#     # Debugging settings -> max precision, set seed
#     dtype = torch.float16 # torch.float16 leads to numerical differences
#     torch.set_float32_matmul_precision("highest")
#     device = "cuda"
#     torch.manual_seed(0)

#     #NOTE: TMA block sizes must be 128 byte aligned
#     # Shapes
#     bs = 1
#     seqlen = 128
#     hl = 4  # Filter size
#     d = 128
#     g = 2
#     dg = d // g
#     # Only for debugging
#     chunk_size = 32 #seqlen // 4
#     BLOCK_D = 32 #if dg > 32 else dg

#     x = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     B = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     C = torch.randn(bs, seqlen, g, dg, device=device, dtype=dtype)
#     h = torch.arange(g * hl, device=device, dtype=dtype).reshape(g, 1, hl)

#     y_ref = gcg_fwd_ref_corrected(x, B, C, h)
#     h_ = h.flip(-1)
#     T_ref = toeplitz(h_[:, 0], chunk_size)
#     T_hat_ref = correction_toeplitz(h_[:, 0], chunk_size)


#     #print(f"{target=} {NUM_SM=} {NUM_REGS=} {SIZE_SMEM=} {WARP_SIZE=}")
#     num_warps = 4
#     swizzle = "row"
#     warmup = False
#     autotune = False
#     schedule = "persistent"
#     return_kernel = False
#     return_autotune_result = False
#     return_toeplitz = True
#     return_y2 = True
#     DEBUG = True

#     if warmup:
#         y = torch.empty_like(x)
#     else:
#         y = None

#     kernel_config = {
#         "CHUNK_SIZE": chunk_size,
#         "BLOCK_D": BLOCK_D,
#         "num_warps": num_warps,
#         "NUM_PIPELINE_STAGES": 0,
#         "num_stages": 2,
#         "THREADBLOCK_SWIZZLE": swizzle,
#         # "DEBUG": DEBUG,
#     }
#     # out_fwd: FwdKernelResult = two_pass_fwd_grouped(
#     #     x,
#     #     B,
#     #     C,
#     #     h,
#     #     y=y,
#     #     autotune=autotune,
#     #     schedule="persistent",
#     #     **kernel_config,
#     #     warmup=warmup,
#     #     return_kernel=return_kernel,
#     #     return_autotune_result=return_autotune_result,
#     #     return_toeplitz=return_toeplitz,
#     #     return_y2=return_y2,
#     #     verbose=False,
#     # )

#     out: FwdKernelResult = two_pass_fwd_grouped_tma(
#         x,
#         B,
#         C,
#         h,
#         y=y,
#         autotune=autotune,
#         schedule="persistent",
#         **kernel_config,
#         warmup=warmup,
#         return_kernel=return_kernel,
#         return_autotune_result=return_autotune_result,
#         return_toeplitz=return_toeplitz,
#         return_y2=return_y2,
#         verbose=True,
#     )
#     y, T, T_hat, y2 = out.y, out.T, out.T_hat, out.y2

#     T_diff = (T - T_ref).abs().max()
#     T_hat_diff = (T_hat - T_hat_ref).abs().max()
#     print(f"{T_diff=}")
#     print(f"{T_hat_diff=}")
#     y_ref_diff = (out.y - y_ref).abs().max()
#     print(f"{y_ref_diff=}")
#     # print(f"y_ref:\n{y_ref}")
#     # print(f"y:\n{out.y}")
#     # y_diff = (out.y - out_fwd.y).abs().max()
#     # print(f"{y_diff=}")
#     # y2_diff = (out.y2 - out_fwd.y2).abs().max()
#     # print(f"{y2_diff=}")
#     # fwd_ref_diff = (out_fwd.y - y_ref).abs().max()
#     # print(f"{fwd_ref_diff=}")
