import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from savanna import mpu, print_rank_0
from savanna.model.operators.local.base import ParallelGLU, ParallelLinear, ParallelMLP
from savanna.mpu.initialize import get_fp8_sync_group

from .operators.local.norms import get_norm

try:
    import transformer_engine.pytorch as te

    from savanna.model.tengine import set_format_recipe
except:
    te = None
    print("WARNING: transformer_engine not installed. Using default recipe.")

try:
    from causal_conv1d import causal_conv1d_fn
except:
    causal_conv1d_fn = None
    print("no custom causal 1d conv installed, using default.")


from savanna.dtype import get_dtype_from_string
from savanna.linear import FlexLinear
from savanna.logging import tb_wandb_log
from savanna.model.operators.fused_bias_dropout import (
    bias_dropout_add_fused_inference,
    bias_dropout_add_fused_train,
    get_bias_dropout_add,
)
from savanna.model.operators.hyena.hyena import ParallelCausalDepthwiseConv1d
from savanna.model.operators.positional_embeddings import (
    AliBi,
    LinearlyScaledRotaryEmbedding,
    NTKScaledRotaryEmbedding,
    RotaryEmbedding,
    RotaryEmbedding2,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
)
from savanna.model.utils import configure_sparse_attention, exists

try:
    from transformer_engine.pytorch.attention import DotProductAttention

    use_te_attention = True
except ImportError:
    DotProductAttention = None
    use_te_attention = False
    print("WARNING: transformer-engine not installed. TE attention disabled.")


def log_norm(norm, key, iteration_no):
    if norm is not None:
        tb_wandb_log(
            key,
            norm,
            iteration_no,
            use_wandb=True,
            tensorboard_writer=None,
            all_ranks=False,
        )


def log_rank_0(x, config, name, layer_number):
    if torch.distributed.get_rank() == 0:
        with torch.no_grad():
            log_norm(
                x.norm(2, dim=-1).max(1).values.mean(0),
                f"{name}_{layer_number}",
                config.iteration,
            )


class ParallelSequenceMixer(nn.Module):
    def __init__(
        self,
        global_config,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
        operator_type=None,
        is_mlp=False,
    ):
        super().__init__()

        self.fp16 = global_config.precision == "fp16"
        self.bf16 = global_config.precision == "bfloat16"
        self.dtype = get_dtype_from_string(global_config.precision)
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = global_config.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = global_config.attention_softmax_in_fp32
        self.is_mlp = is_mlp
        self.global_config = global_config
        self.seq_dim = global_config.seq_dim
        self.interleave_projections = global_config.interleave_projections
        # enable manual setting of attention type
        if operator_type is None:
            self.operator_type = global_config.operator_config[layer_number]
        else:
            self.operator_type = operator_type
        self.grouped_attention = global_config.grouped_attention
        self.use_hyena = self.operator_type in [
            "hyena",
            "hyena_v2",
            "hyena_jumpy",
            "hyena_outer",
            "hyena_short_conv",
            "hyena_medium_conv",
            "hyena_downsample_2",
            "hyena_downsample_4",
            "hyena_downsample_8",
            "hyena_downsample_16",
            "hyena_downsample_32",
        ]
        self.use_flash_attention = self.operator_type == "flash"
        self.use_flash_attention_v2 = self.operator_type == "flash_v2"
        
        if self.operator_type == "flash_te":
            assert global_config.use_cp_flash_te, "use_cp_flash_te must be True when operator_type is flash_te"
        if self.operator_type == "ring":
            assert global_config.use_cp_ring, "use_cp_ring must be True when operator_type is ring"
        if self.operator_type == "flash_v2" and global_config.context_parallel_size > 1:
            assert not (global_config.use_cp_flash_te or global_config.use_cp_ring), "flash_v2 cannot be used with cp_flash_te or cp_ring"

        self.use_te_attention = self.operator_type == "flash_te" and global_config.use_cp_flash_te
        self.use_ring_attention = self.operator_type == "ring" and global_config.use_cp_ring
        self.recycle_events = global_config.recycle_events
        self.use_cp_hyena = "hyena" in self.operator_type and global_config.use_cp_hyena
        
        # print_rank_0(
        #     f"DEBUG::ParallelSequenceMixer::Layer{layer_number}::operator_type: {self.operator_type} {self.use_cp_hyena=} {self.use_flash_attention_v2=} {self.use_te_attention=} {self.use_ring_attention=} {global_config.seq_length=} {mpu.get_sequence_parallel_world_size()=} {global_config.sequence_parallel=}"
        # )
        
        self.sparse = self.operator_type not in (
            "global",
            "flash",
            "flash_v2",
            "hyena",
            "hyena_v2",
            "hyena_jumpy",
            "hyena_outer",
            "hyena_short_conv",
            "hyena_medium_conv",
            "hyena_downsample_2",
            "hyena_downsample_4",
            "hyena_downsample_8",
            "hyena_downsample_16",
            "hyena_downsample_32",
            "ring",
            "flash_te",
        )
        self.causal = global_config.causal

        self.fast_conv_proj = global_config.fast_conv_proj
        self.fast_conv_mixer = global_config.fast_conv_mixer

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()

        # Width expansion for Hyena depending on if it's a mixer of mlp
        if self.is_mlp:
            self.hyena_width_expansion = global_config.hyena_mlp_expansion_factor
        else:
            self.hyena_width_expansion = global_config.hyena_width_expansion

        # we might expand the hidden size for hyena
        self.input_size = global_config.hidden_size
        self.hidden_size = int(
            global_config.hidden_size * self.hyena_width_expansion
            if self.use_hyena
            else global_config.hidden_size
        )

        # This is because hyena_v2 does not use a dense input projection. This is a ColumnParallelLinear
        # which is currently responsible for all-gathering from the sequence parallel region.
        assert not (
            self.operator_type == "hyena_v2" and self.global_config.sequence_parallel
        ), "Sequence parallel not yet supported for hyena_v2"

        # ensures parallizable
        if self.hyena_width_expansion > 1:
            multiple_of = 32
            self.hidden_size = int(multiple_of * ((self.hidden_size + multiple_of - 1) // multiple_of))

        # checks on the hidden size divisibility
        assert (
            self.hidden_size % global_config.num_attention_heads == 0
        ), f"Hidden size {self.hidden_size} is not divisible by the number of attention heads {global_config.num_attention_heads}"
        assert (
            self.hidden_size % world_size == 0
        ), f"Hidden size {self.hidden_size} is not divisible by the world size {world_size}"
        self.hidden_size_per_partition = mpu.divide(self.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(self.hidden_size, global_config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(global_config.num_attention_heads, world_size)
        self.proj_groups = global_config.proj_groups

        self.pos_emb = global_config.pos_emb if not self.use_hyena else None

        self.tie_projection_weights = global_config.tie_projection_weights

        self.grouped_proj_size = global_config.hidden_size // global_config.proj_groups

        # Strided linear layer.
        if (
            self.operator_type != "hyena"
            and self.operator_type != "hyena_short_conv"
            and self.operator_type != "hyena_medium_conv"
        ):
            if self.tie_projection_weights:
                # we'll repeat the output 3 times instead
                projections_size = self.hidden_size
            else:
                self.grouped_proj_size = self.hidden_size // global_config.proj_groups
                projections_size = self.hidden_size + 2 * self.grouped_proj_size
        elif self.tie_projection_weights:
            # we'll repeat the output 3 times instead
            projections_size = self.hidden_size
        else:
            projections_size = 3 * self.hidden_size

        self.use_fp8_input_projections = (
            global_config.use_fp8_hyena_mlp_input_projections
            if is_mlp
            else global_config.use_fp8_input_projections
        )
        extra_kwargs = {} if self.use_fp8_input_projections else {"seq_dim": self.seq_dim}

        # qkv projections
        self.dense_projection = FlexLinear(
            input_size=self.input_size,
            output_size=projections_size,
            bias=False,
            gather_output=False,
            parallel_mode="column",
            init_method=init_method,
            config=global_config,
            use_fp8=self.use_fp8_input_projections,
            **extra_kwargs,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if global_config.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                global_config.num_attention_heads,
                global_config.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from global_config
        if rotary and not self.use_hyena:
            if global_config.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert global_config.rotary_pct < 1
                self.rotary_ndims = int(self.hidden_size_per_attention_head * global_config.rotary_pct)
            dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
            # options for rotary embeddings
            if self.pos_emb == "rotary":  # original
                self.rotary_emb = RotaryEmbedding(
                    dim,
                    base=global_config.rotary_emb_base,
                    precision=global_config.params_dtype,
                )
            elif self.pos_emb == "rotary2":  # similar, but new style
                self.rotary_emb = RotaryEmbedding2(
                    dim,
                    base=global_config.rotary_emb_base,
                    precision=global_config.params_dtype,
                )
            elif self.pos_emb == "rotary_linear_scaled":
                self.rotary_emb = LinearlyScaledRotaryEmbedding(
                    dim,
                    scaling_factor=global_config.rotary_emb_scaling_factor,
                    base=global_config.rotary_emb_base,
                    precision=global_config.params_dtype,
                )
            elif self.pos_emb == "rotary_ntk_scaled":
                self.rotary_emb = NTKScaledRotaryEmbedding(
                    dim,
                    scaling_factor=global_config.rotary_emb_scaling_factor,
                    max_unscaled_seq_len=global_config.rotary_emb_max_unscaled_seq_len,
                    base=global_config.rotary_emb_base,
                    precision=global_config.params_dtype,
                )
            else:
                raise ValueError(f"Unknown rotary positional embedding type {self.pos_emb}")

        else:
            self.rotary_emb = None

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.dropout_p = global_config.attention_dropout
        self.attention_dropout = nn.Dropout(self.dropout_p)

        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                global_config,
                self.operator_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:

            if self.operator_type == "hyena_short_conv":
                from savanna.model.operators.hyena.hyena import (
                    ParallelShortHyenaOperator,
                )

                ## updating for model parallel
                hyena_proj_groups = global_config.proj_groups if not self.grouped_attention else 1
                self.num_groups = global_config.num_groups_hyena_short
                self.num_groups_per_tp_rank = self.num_groups // global_config.model_parallel_size
                grouped_proj_size = self.hidden_size_per_partition // hyena_proj_groups

                self.hyena_proj_conv = ParallelCausalDepthwiseConv1d(
                    self.hidden_size_per_partition + 2 * grouped_proj_size,
                    global_config=global_config,
                    kernel_size=global_config.short_conv_L,  # or 5
                    init_method=init_method,
                    bias=global_config.conv_proj_bias,
                    use_fast_causal_conv=self.fast_conv_proj,
                )

                short_conv_class = ParallelCausalDepthwiseConv1d

                self.mixer = ParallelShortHyenaOperator(
                    self.hidden_size,  # pass hidden size here to avoid recalculating
                    global_config,
                    init_method,
                    short_conv_class=short_conv_class,
                    use_fast_causal_conv=self.fast_conv_mixer,
                    is_mlp=self.is_mlp,
                )

            elif self.use_hyena:
                if self.operator_type in [
                    "hyena",
                    "hyena_medium_conv",
                    "hyena_downsample_2",
                    "hyena_downsample_4",
                    "hyena_downsample_8",
                    "hyena_downsample_16",
                    "hyena_downsample_32",
                ]:
                    from savanna.model.operators.hyena.hyena import (
                        ParallelHyenaOperator,
                    )

                    hyena_proj_groups = global_config.proj_groups if not self.grouped_attention else 1
                    grouped_proj_size = self.hidden_size_per_partition // hyena_proj_groups
                    self.num_groups = (
                        global_config.num_groups_hyena_medium
                        if self.operator_type == "hyena_medium_conv"
                        else global_config.num_groups_hyena
                    )
                    self.num_groups_per_tp_rank = self.num_groups // global_config.model_parallel_size

                    self.hyena_proj_conv = ParallelCausalDepthwiseConv1d(
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        global_config=global_config,
                        kernel_size=global_config.short_conv_L,
                        init_method=init_method,
                        bias=global_config.conv_proj_bias,
                        use_fast_causal_conv=self.fast_conv_proj,
                    )

                    downsample_factor = 1
                    if "downsample" in self.operator_type:
                        downsample_factor = int(self.operator_type.split("_")[-1])

                    self.mixer = ParallelHyenaOperator(
                        self.hidden_size,  # pass hidden size here to avoid recalculating
                        global_config,
                        init_method,
                        layer_number,
                        downsample_factor=downsample_factor,
                    )
                elif self.operator_type == "hyena_v2":
                    from savanna.model.operators.hyena.hyena import (
                        ParallelHyenaOperator2,
                    )

                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition,
                        3 * self.hidden_size_per_partition,
                        global_config.short_conv_L,
                        padding=global_config.short_conv_L - 1,
                        bias=True,
                    )

                    self.mixer = ParallelHyenaOperator2(global_config, init_method, layer_number)
                elif self.operator_type == "hyena_jumpy":
                    from savanna.model.operators.hyena.hyena import (
                        ParallelHyenaOperatorJumpy,
                    )

                    grouped_proj_size = global_config.hidden_size // global_config.proj_groups
                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        global_config.short_conv_L,
                        groups=self.hidden_size_per_partition + 2 * grouped_proj_size,
                        padding=global_config.short_conv_L - 1,
                    )
                    self.mixer = ParallelHyenaOperatorJumpy(global_config, init_method, layer_number)
                elif self.operator_type == "hyena_outer":
                    from savanna.model.operators.hyena.hyena import (
                        ParallelHyenaOperator2,
                    )

                    grouped_proj_size = global_config.hidden_size // global_config.proj_groups
                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        global_config.short_conv_L,
                        groups=self.hidden_size_per_partition + 2 * grouped_proj_size,
                        padding=global_config.short_conv_L - 1,
                    )
                    self.mixer = ParallelHyenaOperator2(global_config, init_method, layer_number)
                    
            elif self.use_flash_attention:
                from savanna.model.operators.attention.flash import (
                    flash_attn_unpadded_kvpacked_func_cuda,
                    flash_attn_unpadded_qkvpacked_func_cuda,
                    flash_attn_unpadded_unpacked_func_triton,
                )

                self.flash_triton_fn = flash_attn_unpadded_unpacked_func_triton
                self.flash_qkv_fn = flash_attn_unpadded_qkvpacked_func_cuda
                self.flash_kv_fn = flash_attn_unpadded_kvpacked_func_cuda

            elif self.use_flash_attention_v2:
                from savanna.model.operators.attention.flash import FlashSelfAttention

                self.attn = FlashSelfAttention(causal=self.causal)

            elif self.use_te_attention:
                # raise ValueError("Flash TE is not tested to be working correctly yet")
                from transformer_engine.pytorch.attention import DotProductAttention

                from savanna.mpu import get_cuda_rng_tracker, get_model_parallel_group

                if mpu.get_sequence_parallel_world_size() > 1:
                    # CP activated
                    cp_group = mpu.get_sequence_parallel_group()
                    cp_global_ranks = dist.get_process_group_ranks(cp_group)
                    cp_stream = torch.cuda.Stream()
                else:
                    cp_group = None
                    cp_global_ranks = None
                    cp_stream = None

                cp_comm_type = global_config.flash_te_comm_type
                assert cp_comm_type in ["p2p", "all_gather", "a2a"]
                #NOTE: @jeromeku: IMPORTANT: need to pass num_attention_heads and NOT num_attention_heads_per_partition, as DotProductAttention will calculate num_attention_heads_per_partition from num_attention_heads and tp_size
                #will result in assertion error if we pass num_attention_heads_per_partition since DotProductAttention will check in `forward`:
                #                 key_layer.shape[-2] == self.num_gqa_groups_per_partition
                #                 and value_layer.shape[-2] == self.num_gqa_groups_per_partition
                # where self.num_gqa_groups_per_partition is calculated as num_attention_heads / tp_size
                

                # print_rank_0(
                #     f"DEBUG::ParallelSequenceMixer::use_te_attention: {mpu.get_sequence_parallel_world_size()=} {mpu.get_model_parallel_world_size()=} {self.global_config.flash_te_comm_type=} {self.global_config.te_attn_backend=} {self.num_attention_heads_per_partition=} {self.hidden_size_per_attention_head=}"
                # )
                self.attn = DotProductAttention(
                    num_attention_heads=self.global_config.num_attention_heads, # NOTE: @jeromeku: IMPORTANT: pass num_attention_heads and NOT num_attention_heads_per_partition
                    kv_channels=self.hidden_size_per_attention_head,
                    num_gqa_groups=None,
                    attention_dropout=self.dropout_p,
                    qkv_format="sbhd",
                    attn_mask_type="causal",
                    window_size=None,
                    sequence_parallel=global_config.sequence_parallel,
                    tp_group=get_model_parallel_group(check_initialized=False),
                    tp_size=global_config.model_parallel_size,
                    get_rng_state_tracker=get_cuda_rng_tracker,
                    layer_number=None,
                    attention_type="self",
                    cp_group=cp_group,
                    cp_global_ranks=cp_global_ranks,
                    cp_stream=cp_stream,
                    cp_comm_type=cp_comm_type, #"p2p","all-gather", "a2a"
                    softmax_scale=None,
                )

            elif self.use_ring_attention:
                from savanna.model.operators.attention.flash import RingAttention

                # print_rank_0(f"DEBUG::ParallelSequenceMixer::use_ring_attention: {self.num_attention_heads_per_partition=} {self.hidden_size_per_attention_head=}")
                self.attn = RingAttention(
                    global_config=global_config,
                    causal=self.causal,
                    num_attention_heads_per_partition=self.num_attention_heads_per_partition,
                    hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                    pos_emb=self.pos_emb,
                    alibi_embed=self.alibi_embed if self.pos_emb == "alibi" else None,
                    dropout_p=self.dropout_p,
                )
                # if self.recycle_events:
                #     print_rank_0(f"DEBUG::ParallelSequenceMixer::ring_attention:recycle_events: {self.global_config.recycle_events=} disabling torch.compile")
                #     self.attn = torch.compiler.disable(self.attn)
                # from ring_flash_attn.zigzag_ring_flash_attn_varlen import (
                #     zigzag_ring_flash_attn_varlen_qkvpacked_func,
                # )
                # print_rank_0(f"DEBUG::ParallelSequenceMixer::use_ring_attention: {mpu.get_sequence_parallel_world_size()=}")
                # self.attn = self.ring_attn_fn = zigzag_ring_flash_attn_varlen_qkvpacked_func
            else:
                raise ValueError("Unknown attention type")

        self.use_fp8_output_projections = (
            global_config.use_fp8_hyena_mlp_output_projections
            if is_mlp
            else global_config.use_fp8_output_projections
        )

        extra_kwargs = {} if self.use_fp8_output_projections else {"seq_dim": self.seq_dim}

        self.dense = FlexLinear(
            input_size=self.hidden_size,
            output_size=self.input_size,
            bias=True,
            config=global_config,
            parallel_mode="row",
            skip_bias_add=True,
            input_is_parallel=True,
            parallel_output=parallel_output,
            init_method=output_layer_init_method,
            use_fp8=self.use_fp8_output_projections,
            **extra_kwargs,
        )

        if self.use_fp8_input_projections or self.use_fp8_output_projections:
            self.fp8_format, self.fp8_recipe = set_format_recipe(global_config)

    def flash_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
    ):
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        batch_size = output_size[0]
        max_seqlen_q = output_size[2]
        max_seqlen_k = output_size[3]

        # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
        key_layer = key_layer.transpose(0, 1).reshape(output_size[0] * output_size[3], 1, output_size[1], -1)
        value_layer = value_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )

        # [sq, b, np, hn] -> [b * sq, 1, np, hn]
        query_layer = query_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[2], 1, output_size[1], -1
        )

        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_q,
            step=max_seqlen_q,
            dtype=torch.int32,
            device=query_layer.device,
        )

        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_k,
            step=max_seqlen_k,
            dtype=torch.int32,
            device=key_layer.device,
        )

        # Combined q/k/v into [b * s, 3, np, hn].
        qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

        if self.operator_type == "flash":
            output = self.flash_qkv_fn(
                qkv,
                cu_seqlens_q,
                max_seqlen_q,
                self.dropout_p if self.training else 0.0,
                softmax_scale=None,
                causal=self.causal,
            )

        elif self.operator_type == "flash_v2":
            output = self.attn(
                qkv,
                self.causal,
                cu_seqlens_q,
                max_seqlen_q,
            )

        output = rearrange(
            output,
            "(b l) h dh -> b l (h dh)",
            b=batch_size,
            l=max_seqlen_q,
            h=self.num_attention_heads_per_partition,
            dh=self.hidden_size_per_attention_head,
        )
        return output

        # [b * sq, np, hn] -> [b, sq, np, hn]
        # matmul_result = output.view(output_size[0], output_size[2], output.shape[1], output.shape[2])
        # [b, sq, np, hn] -> [b, np, sq, hn]
        # matmul_result = matmul_result.transpose(1, 2)

        # return matmul_result

    def te_attention(
            self,
            query_layer,  # Shapes are [seq, batch, num_head, head_dim]
            key_layer,  # Shapes are [seq, batch, num_head, head_dim]
            value_layer,  # Shapes are [seq, batch, num_head, head_dim]
    ):
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        batch_size = output_size[0]
        max_seqlen_q = output_size[2]
        max_seqlen_k = output_size[3]

        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_q,
            step=max_seqlen_q,
            dtype=torch.int32,
            device=query_layer.device,
        )

        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_k,
            step=max_seqlen_k,
            dtype=torch.int32,
            device=key_layer.device,
        )

        # only pass in alibi_slopes kwarg] if we use AliBi.
        # Flash attn defaults to (-1,-1), or
        # does not have this kwarg prior to v2.3.0
        extra_kwargs = {}
        if self.pos_emb == "alibi":
            extra_kwargs["alibi_slopes"] = self.alibi_embed.slopes.to(
                query_layer.device
            ).to(torch.float32)

        output = self.attn(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            # cu_seqlens_kv=cu_seqlens_k,
            # max_seqlen_kv=max_seqlen_k,
            **extra_kwargs,
        )

        output = rearrange(output, "l b (h dh) -> b l (h dh)",
                           b=batch_size, l=max_seqlen_q,
                           h=self.num_attention_heads_per_partition, dh=self.hidden_size_per_attention_head)
        
        # output = output.permute(1, 0, 2) # [L, B, D] -> [B, L, D]
        
        return output


    # def ring_attention(self, query_layer, key_layer, value_layer):
    #     # [b, np, sq, sk]
    #     output_size = (
    #         query_layer.size(1),
    #         query_layer.size(2),
    #         query_layer.size(0),
    #         key_layer.size(0),
    #     )
    #     batch_size = output_size[0]
    #     max_seqlen_q = output_size[2]
    #     max_seqlen_k = output_size[3]

    #     # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
    #     key_layer = key_layer.transpose(0, 1).reshape(output_size[0] * output_size[3], 1, output_size[1], -1)
    #     value_layer = value_layer.transpose(0, 1).reshape(
    #         output_size[0] * output_size[3], 1, output_size[1], -1
    #     )

    #     # [sq, b, np, hn] -> [b * sq, 1, np, hn]
    #     query_layer = query_layer.transpose(0, 1).reshape(
    #         output_size[0] * output_size[2], 1, output_size[1], -1
    #     )

    #     cu_seqlens_q = torch.arange(
    #         0,
    #         (batch_size + 1) * max_seqlen_q,
    #         step=max_seqlen_q,
    #         dtype=torch.int32,
    #         device=query_layer.device,
    #     )

    #     # Combined q/k/v into [b * s, 3, np, hn].
    #     qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

    #     # only pass in alibi_slopes kwarg] if we use AliBi.
    #     # Flash attn defaults to (-1,-1), or
    #     # does not have this kwarg prior to v2.3.0
    #     extra_kwargs = {}
    #     if self.pos_emb == "alibi":
    #         extra_kwargs["alibi_slopes"] = self.alibi_embed.slopes.to(query_layer.device).to(torch.float32)

    #     output = self.attn(
    #         qkv,
    #         cu_seqlens_q,
    #         max_seqlen_q,
    #         self.dropout_p if self.training else 0.0,
    #         softmax_scale=None,
    #         causal=self.causal,
    #         group=mpu.get_sequence_parallel_group(),
    #         **extra_kwargs,
    #     )

    #     output = rearrange(
    #         output,
    #         "(b l) h dh -> b l (h dh)",
    #         b=batch_size,
    #         l=max_seqlen_q,
    #         h=self.num_attention_heads_per_partition,
    #         dh=self.hidden_size_per_attention_head,
    #     )
    #     return output

    def ring_attn(self, query_layer, key_layer, value_layer):
        return self.attn(query_layer, key_layer, value_layer)

    def split_projections(self, x, proj_groups=1):
        if proj_groups == 1:
            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = x.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            x = x.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (x1, x2, value) = mpu.split_tensor_along_last_dim(x, 3)
        else:
            grouped_projs = x[
                ...,
                self.hidden_size_per_attention_head * self.num_attention_heads_per_partition :,
            ]
            grouped_projs = grouped_projs.repeat(1, 1, proj_groups)
            x = torch.cat(
                [
                    x[
                        ...,
                        : self.hidden_size_per_attention_head * self.num_attention_heads_per_partition,
                    ],
                    grouped_projs,
                ],
                dim=-1,
            )

            new_tensor_shape = x.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            x = x.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (x1, x2, value) = mpu.split_tensor_along_last_dim(x, 3)

        return x1, x2, value

    def forward(self, u, layer_past=None):
        """
        Applies sequence mixing to a sequence of 1-dimensional embeddings: batch_size, seq_len, d_model

        Args:
            u: input to the operator, in format [B, L, D]
        """
        B, L, D = u.size()
        _B, _L, _D = B, L, D
       # print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:{self.operator_type}::u {B=} {L=} {D=} {self.global_config.seq_length=} {mpu.get_sequence_parallel_world_size()=} {self.global_config.sequence_parallel=}")
        
        if self.use_fp8_input_projections:
            # @jeromeku - Permute so that L is the first dimension per TE requirements for SP
            # u shape: [B, L, D] -> [L, B, D]
            if self.global_config.sequence_parallel:
                u = u.permute(1, 0, 2).contiguous()
            # print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:{self.operator_type}::u_permute {u.shape=}")

            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe, fp8_group=get_fp8_sync_group()):
                features, _ = self.dense_projection(u)
            
            # print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:{self.operator_type}::features {features.shape=}")
            
            # @jeromeku - Permute back to original shape
            # features shape: [L, B, D] -> [B, L, D]
            # NOTE: @jeromeku, when using flash or ring attention, we do not need to permute, as desired post-input projection input shape prior to qkv splitting is[L, B, D]
            if self.global_config.sequence_parallel:
                if not self.operator_type in ["flash", "flash_v2", "ring", "flash_te"]:
                    features = features.permute(1, 0, 2).contiguous()
                    B, L, D = features.size()
                else:
                    L, B, D = features.size()
            else:
                B, L, D = features.size()
        else:
            features, _ = self.dense_projection(u)
            B, L, D = features.size()


        #print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:features_after_dense {features.shape=}")
        # NOTE: @jeromeku, reset B, L, D here AFTER dense_projection AG to account for SP: L -> L * MP == seq_length
        # Since dense_projection is a column parallel linear layer, it will split hidden dimension D into D // MP, where D = 3 * _D (qkv projection)
        # B should not change
        prefix = f"Shape mismatch after ParallelSequenceMixer:{self.layer_number}:dense_projection"
        expected_D = _D * 3 // self.global_config.model_parallel_size
        assert B == _B, f"{prefix}: {B=} != {_B=}"
        assert D == expected_D, f"{prefix}: {D=} != {expected_D=}"

        seq_len = self.global_config.seq_length // mpu.get_sequence_parallel_world_size()
        # print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:{self.operator_type}::seq_len {seq_len} = {self.global_config.seq_length=} // {mpu.get_sequence_parallel_world_size()=}")

        if not self.global_config.enable_haystack:
            if self.global_config.sequence_parallel:
                pass
                # expected_L = _L * self.global_config.model_parallel_size
                # print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:{self.operator_type}::sequence_parallel::expected_L {expected_L} = {_L=} * {self.global_config.model_parallel_size=}")
                # assert L == expected_L == seq_len, f"{prefix}: {L=} != {expected_L=} != {seq_len=}"
            else:
                pass
                # print_rank_0(f"DEBUG::ParallelSequenceMixer::forward:{self.operator_type}::sequence_parallel_false::expected_L {expected_L} = {_L=} * {self.global_config.model_parallel_size=}")
                # assert L == _L == seq_len, f"{prefix}: {L=} != {_L=} != {seq_len=}"

        if self.operator_type in ["hyena", "hyena_short_conv", "hyena_medium_conv"]:

            cp_group = mpu.get_sequence_parallel_group()

            features_L_last = features.permute(0, 2, 1)

            # @jeromeku: IMPORTANT: when using SP, `L = is seq_length // MP` thus truncating by L below is incorrect and is in fact NOT needed
            # for evo2 models that use ParallelDepthwiseConv1D, which already truncates by the **correct** L within `forward`
            features_D_last = self.hyena_proj_conv(features_L_last)
            features_D_last = features_D_last[..., :L]
            features_D_last = features_D_last.permute(0, 2, 1)

            # Should almost always interleave, only case is if for some reason continuing to pre-train a legacy evo2 checkpoint
            # that was incorrectly trained without interleaving at a given MP size.
            if not self.interleave_projections:
                x1, x2, v = rearrange(
                    features_D_last, "b l (p g dg) -> b l p g dg", p=3, g=self.num_groups_per_tp_rank
                ).unbind(dim=2)
            else:
                # Interleaved Splitting
                # Vectorized implementation h/t @dwromero-nv
                x1, x2, v = rearrange(
                    features_D_last, "b l (g dg p) -> b l g p dg", p=3, g=self.num_groups_per_tp_rank
                ).unbind(dim=3)

            z = self.mixer(x1, x2, v)

            if self.use_fp8_output_projections:
                # @jeromeku - Permute so that L is the first dimension per TE requirements for SP
                # z shape: [B, L, D] -> [L, B, D]
                if self.global_config.sequence_parallel:
                    z = z.permute(1, 0, 2).contiguous()

                with te.fp8_autocast(
                    enabled=True, fp8_recipe=self.fp8_recipe, fp8_group=get_fp8_sync_group()
                ):
                    y, bias = self.dense(z)

                # @jeromeku - Permute back to original shape
                # y shape: [L, B, D] -> [B, L, D]
                if self.global_config.sequence_parallel:
                    y = y.permute(1, 0, 2).contiguous()

            else:
                y, bias = self.dense(z)

            # @jeromeku if using SP, y should be scattered along seq_len dimension
            B, L, D = y.shape
            prefix = f"Shape mismatch after ParallelSequenceMixer:{self.layer_number}:dense:out_proj"
            assert L == y.shape[self.seq_dim], f"{prefix}: {L=} != {y.shape[self.seq_dim]=}"
            seq_len = self.global_config.seq_length // mpu.get_sequence_parallel_world_size()

            if not self.global_config.enable_haystack:  
                if self.global_config.sequence_parallel:
                    pass
                    # assert (
                    #     L == seq_len // self.global_config.model_parallel_size
                    # ), f"{prefix}: {L=} != {seq_len // self.global_config.model_parallel_size=}"
                else:
                    pass
                    # assert L == seq_len, f"{prefix}: {L=} != {seq_len=}"

            return y, bias
        else:
 
            # if sharing projection weights need to repeat 3 times and concat along channel
            if self.tie_projection_weights:
                mixed_x_layer = mixed_x_layer.repeat_interleave(3, dim=-1)

            # TODO: consolidate with L last after new refactor
            # B, L, D -> L, B, D
            # NOTE: @jeromeku, when using sequence parallel and fp8, we do not need to permute, as output of the fp8 input projections is already [L, B, D]
 
            if not (self.global_config.sequence_parallel and self.use_fp8_input_projections):
                features = features.permute(1, 0, 2)
 
            q, k, v = self.split_projections(features, self.proj_groups)
 
            if exists(self.rotary_emb):
                if exists(self.rotary_ndims):
                    # partial rotary
                    query_rot, query_pass = (
                        q[..., : self.rotary_ndims],
                        q[..., self.rotary_ndims :],
                    )
                    key_rot, key_pass = (
                        k[..., : self.rotary_ndims],
                        k[..., self.rotary_ndims :],
                    )
                else:
                    # full rotary
                    query_rot, key_rot = q, k
                apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

                seq_len = k.shape[0]
                offset = 0
                cos, sin = self.rotary_emb(v, seq_len=seq_len)
                q, k = apply_rotary_fn(query_rot, key_rot, cos, sin, offset=offset)

                if exists(self.rotary_ndims):
                    q = torch.cat((q, query_pass), dim=-1)
                    k = torch.cat((k, key_pass), dim=-1)

            if self.use_flash_attention or self.use_flash_attention_v2:
                z = self.flash_attention(q, k, v)
            elif self.use_te_attention:
                z = self.te_attention(q, k, v)
            else:
                z = self.ring_attn(q, k, v)

            if self.use_fp8_output_projections:
                # @jeromeku - Permute so that L is the first dimension per TE requirements for SP
                # z shape: [B, L, D] -> [L, B, D]

                if self.global_config.sequence_parallel:
                    z = z.permute(1, 0, 2).contiguous()

                with te.fp8_autocast(
                    enabled=True, fp8_recipe=self.fp8_recipe, fp8_group=get_fp8_sync_group()
                ):
                    y, bias = self.dense(z)

                # @jeromeku - Permute back to original shape
                # y shape: [L, B, D] -> [B, L, D]
                if self.global_config.sequence_parallel:
                    y = y.permute(1, 0, 2).contiguous()

            else:
                y, bias = self.dense(z)

            # @jeromeku if using SP, y should be scattered along sequence dimension
            B, L, D = y.shape
            prefix = f"Shape mismatch after ParallelSequenceMixer:{self.layer_number}:dense:out_proj"
            # assert L == y.shape[self.seq_dim], f"{prefix}: {L=} != {y.shape[self.seq_dim]=}"
            seq_len = self.global_config.seq_length // mpu.get_sequence_parallel_world_size()
            
            if not self.global_config.enable_haystack:
                if self.global_config.sequence_parallel:
                    pass
                    # assert (
                    #     L == seq_len // self.global_config.model_parallel_size
                    # ), f"{prefix}: {L=} != {seq_len // self.global_config.model_parallel_size=}"
                else:
                    pass
                    # assert L == seq_len, f"{prefix}: {L=} != {seq_len=}"

            return y, bias


class ParallelBlock(nn.Module):
    def __init__(
        self,
        global_config,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):
        super().__init__()

        self.layer_number = layer_number

        self.log_attn_norms = global_config.log_attn_norms
        self.global_config = global_config

        norm, eps = get_norm(global_config)
        self.prenorm, self.postnorm = global_config.prenorm, global_config.postnorm
        if global_config.prenorm:
            self.input_layernorm = norm(global_config.hidden_size, eps=eps)

        self.operator_type = global_config.operator_config[layer_number]
        # if global_config.postnorm and self.operator_type != "hyena":
        self.post_attention_layernorm = norm(global_config.hidden_size, eps=eps)
        self.pre_mlp_layernorm = norm(global_config.hidden_size, eps=eps)
        self.outer_mlp_layernorm = norm(global_config.hidden_size, eps=eps)

        self.use_cache = use_cache

        self.hidden_dropout = global_config.hidden_dropout
        self.bias_dropout_fusion = global_config.bias_dropout_fusion
        self.gpt_j_residual = global_config.gpt_j_residual
        self.gpt_j_tied = global_config.gpt_j_tied
        self.mlp_type = global_config.mlp_type
        self.pre_mlp_norm = global_config.pre_mlp_norm
        self.outer_mlp_norm = global_config.outer_mlp_norm
        self.safe = global_config.safe

        if self.gpt_j_residual:
            # GPT-J style layers allow us to defer the reduction of results across TP ranks until the end of the two sublayers.
            # the reduction we use is a simple allreduce for pure Tensor Parallel,
            # but needs to be a reduce-scatter when using Megatron-style Sequence Parallel (LN sharding.)
            self.reduce = (
                mpu.mappings.reduce_from_model_parallel_region
                if not global_config.sequence_parallel
                else mpu.mappings.reduce_scatter_to_sequence_parallel_region
            )

        # Self attention.
        self.mixer = ParallelSequenceMixer(
            global_config=global_config,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        # MLP
        if global_config.all_config["identity_mlp"]:
            self.mlp = nn.Identity()
        else:
            if global_config.mlp_type == "regular" or (self.safe and self.operator_type == "hyena"):
                self.mlp = ParallelMLP(
                    global_config=global_config,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                )
            elif global_config.mlp_type == "short_hyena":
                self.mlp = ParallelSequenceMixer(
                    global_config=global_config,
                    attention_mask_func=attention_mask_func,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number,
                    rpe=rpe,
                    use_cache=self.use_cache,
                    rotary=rotary,
                    parallel_output=self.gpt_j_residual,
                    operator_type="hyena_short_conv",
                    is_mlp=True,  # needs to know if it's an mlp in order to check for diff filter len
                )
            else:
                self.mlp = ParallelGLU(
                    global_config=global_config,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                    multiple_of=global_config.make_gated_mlp_multiple_of,
                )

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = bias_dropout_add_fused_train if self.training else bias_dropout_add_fused_inference
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def sequential_residual_forward(
        self,
        x,
        bias_dropout_fn,
    ):
        assert not (self.pre_mlp_norm and self.postnorm), "Pre MLP norm and postnorm are both active."
        residual = x

        if self.prenorm:
            if self.log_attn_norms:
                log_rank_0(x, self.global_config, "prenorm_xnorm")
            x = self.input_layernorm(x)
            if self.log_attn_norms:
                log_rank_0(x, self.global_config, "prenorm_xnorm_post")

        mixer_output, out_proj_bias = self.mixer(x)

        if torch.distributed.get_rank() == 0 and self.log_attn_norms:
            with torch.no_grad():
                log_norm(
                    mixer_output.norm(2, dim=-1).max(1).values.mean(0),
                    f"post_attn_norm_{self.layer_number}",
                    self.global_config.iteration,
                )
                log_norm(
                    out_proj_bias.expand_as(residual).norm(2, dim=-1).max(1).values.mean(0),
                    f"post_attn_residual_bias_borm_{self.layer_number}",
                    self.global_config.iteration,
                )

        with torch.enable_grad():
            mixer_output = bias_dropout_fn(
                mixer_output,
                bias=out_proj_bias,
                residual=residual,
                prob=self.hidden_dropout,
            )

        if torch.distributed.get_rank() == 0 and self.log_attn_norms:
            with torch.no_grad():
                log_norm(
                    mixer_output.norm(2, dim=-1).max(1).values.mean(0),
                    f"post_attn_residual_norm_{self.layer_number}",
                    self.global_config.iteration,
                )

        if isinstance(self.mlp, nn.Identity):
            output = mixer_output

        else:
            # output = x + mlp(ln2(x))
            # option for pre-mlp norm
            if self.pre_mlp_norm:
                output = self.pre_mlp_layernorm(mixer_output)
            else:
                output = mixer_output

            mlp_output, mlp_bias = self.mlp(output)

            if self.global_config.debug_print:
                print_rank_0(f"DEBUG::ParallelBlock::mlp_output: {mlp_output.shape}")

            if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                with torch.no_grad():
                    log_norm(
                        mlp_output.norm(2, dim=-1).max(1).values.mean(0),
                        f"post_mlp_norm_{self.layer_number}",
                        self.global_config.iteration,
                    )

            if self.postnorm:
                mlp_output = self.post_attention_layernorm(mlp_output)

            if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                with torch.no_grad():
                    log_norm(
                        mlp_output.norm(2, dim=-1).max(1).values.mean(0),
                        f"post_postnorm_norm_{self.layer_number}",
                        self.global_config.iteration,
                    )

            if self.mlp_type == "llama" or self.mlp_type == "doublegate_llama":
                output = mixer_output + mlp_output  # attention_output is the residual basically

                # after attn AND residual
                if self.outer_mlp_norm:
                    assert not (
                        self.postnorm and self.outer_mlp_norm
                    ), "can't do both post norm and pre mlp norm!"
                    output = self.outer_mlp_layernorm(output)

            else:
                with torch.enable_grad():
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(mlp_output),
                        residual=mixer_output,
                        prob=self.hidden_dropout,
                    )
        return output

    def forward(self, x):
        bias_dropout_fn = self._get_bias_dropout()

        if type(x) == tuple:
            x = x[0]

        # x: [b, l, d]
        output = self.sequential_residual_forward(
            x,
            bias_dropout_fn,
        )

        return output


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(args, torch.Tensor), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, bias = super().forward(hidden_state)

        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(args, tuple), "NormPipe should only receive a single tensor as input"
        out = self.norm(args)
        return out


def parallel_lm_logits(
    input_, word_embeddings_weight, parallel_output, seq_parallel=False, seq_dim=1, bias=None
):
    """LM logits using word embedding weights."""
    # Parallel logits.
    if seq_parallel:
        # if using Sequence Parallelism, our logits are sharded along the sequence dimension.
        # gather them here. (backward pass: reduce-scatter)
        input_parallel = mpu.gather_from_sequence_parallel_region(input_, seq_dim=seq_dim)
    else:
        # Set up backprop all-reduce.
        input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    out = mpu.gather_from_model_parallel_region(logits_parallel)
    return out
    return out
    return out
    return out
