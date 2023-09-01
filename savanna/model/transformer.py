import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
from savanna import mpu
from savanna.model.fused_softmax import FusedScaleMaskSoftmax
from savanna.model.activations import get_activation
from savanna.model.utils import exists, get_fusion_type
from savanna.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
)
from savanna.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from savanna.model.utils import configure_sparse_attention
from savanna.logging import tb_wandb_log
from savanna.model.hyena import get_dtype_from_string


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


# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        global_config,
        init_method,
        output_layer_init_method,
        parallel_output=False,
    ):
        super().__init__()

        self.activation_func = get_activation(global_config)
        self.activation_type = global_config.activation
        self.bias_gelu_fusion = global_config.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = 4 * 2 / 3 if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * global_config.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * global_config.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            global_config=global_config,
            input_size=global_config.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            global_config=global_config,
            input_size=ff_dim_in,
            output_size=global_config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelGatedMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        global_config,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
    ):
        super().__init__()

        self.activation_func = get_activation(global_config)
        self.activation_type = global_config.activation

        self.multiple_of = multiple_of

        ff_dim = int(2 * global_config.hidden_size * 4 / 3)
        ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        self.w1 = mpu.ColumnParallelLinear(
            global_config=global_config,
            input_size=global_config.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w3 = mpu.ColumnParallelLinear(
            global_config=global_config,
            input_size=global_config.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w2 = mpu.RowParallelLinear(
            global_config=global_config,
            input_size=ff_dim,
            output_size=global_config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        global_config,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = global_config.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                global_config=global_config,
                input_size=global_config.hidden_size,
                output_size=global_config.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if global_config.use_mup = True, despite it not being included here
            )
        else:
            self.final_linear = mpu.RowParallelLinear(
                global_config=global_config,
                input_size=global_config.hidden_size,
                output_size=global_config.padded_vocab_size,
                bias=False,
                input_is_parallel=False,
                init_method=init_method,
                parallel_output=parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # only called if global_config.use_mup = True, despite it not being included here
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

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
    ):
        super().__init__()

        self.fp16 = global_config.precision == "fp16"
        self.bf16 = global_config.precision == "bfloat16"
        self.dtype = get_dtype_from_string(global_config.precision)
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = global_config.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = global_config.attention_softmax_in_fp32
        self.attention_type = global_config.attention_config[layer_number]
        self.grouped_attention = global_config.grouped_attention
        self.use_badger = self.attention_type == "badger"
        self.use_hyena = (
            self.attention_type == "hyena"
            or self.attention_type == "hyena_v2"
            or self.attention_type == "hyena_jumpy"
            or self.attention_type == "hyena_outer"
        )
        self.use_flash_attention = self.attention_type == "flash"
        self.use_flash_attention_v2 = self.attention_type == "flash_v2"
        self.sparse = self.attention_type not in (
            "global",
            "flash",
            "flash_v2",
            "h3",
            "badger",
            "hyena",
            "hyena_v2",
            "hyena_jumpy",
            "hyena_outer",
        )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size = global_config.hidden_size
        self.hidden_size_per_partition = mpu.divide(
            global_config.hidden_size, world_size
        )
        self.hidden_size_per_attention_head = mpu.divide(
            global_config.hidden_size, global_config.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            global_config.num_attention_heads, world_size
        )
        self.proj_groups = global_config.proj_groups

        self.pos_emb = global_config.pos_emb if not self.use_hyena else None

        # Strided linear layer.
        if self.attention_type != "hyena":
            self.grouped_proj_size = (
                global_config.hidden_size // global_config.proj_groups
            )
            projections_size = global_config.hidden_size + 2 * self.grouped_proj_size
        else:
            projections_size = 3 * global_config.hidden_size

        self.query_key_value = mpu.ColumnParallelLinear(
            global_config=global_config,
            input_size=global_config.hidden_size,
            output_size=projections_size,
            gather_output=False,
            init_method=init_method,
            bias=True,
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
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * global_config.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim,
                base=global_config.rotary_emb_base,
                precision=global_config.params_dtype,
            )
        else:
            self.rotary_emb = None

        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                global_config,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_h3:
                from savanna.model.long_conv import ParallelH3Conv

                self.mixer = ParallelH3Conv(global_config)
            elif self.use_badger:
                from savanna.model.long_conv import ParallelBadgerConv

                self.mixer = ParallelBadgerConv(global_config)
            elif self.use_hyena:
                if self.attention_type == "hyena":
                    from savanna.model.hyena import ParallelHyenaConv

                    hyena_proj_groups = (
                        global_config.proj_groups if not self.grouped_attention else 1
                    )
                    grouped_proj_size = global_config.hidden_size // hyena_proj_groups
                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        global_config.short_conv_L,
                        groups=self.hidden_size_per_partition + 2 * grouped_proj_size,
                        padding=global_config.short_conv_L - 1,
                        bias=True,
                    )

                    self.mixer = ParallelHyenaConv(
                        global_config, init_method, layer_number
                    )
                elif self.attention_type == "hyena_v2":
                    from savanna.model.hyena import ParallelHyenaConv2

                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition,
                        3 * self.hidden_size_per_partition,
                        global_config.short_conv_L,
                        padding=global_config.short_conv_L - 1,
                        bias=True,
                    )

                    self.mixer = ParallelHyenaConv2(
                        global_config, init_method, layer_number
                    )
                elif self.attention_type == "hyena_jumpy":
                    from savanna.model.hyena import ParallelHyenaConvJumpy

                    grouped_proj_size = (
                        global_config.hidden_size // global_config.proj_groups
                    )
                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        global_config.short_conv_L,
                        groups=self.hidden_size_per_partition + 2 * grouped_proj_size,
                        padding=global_config.short_conv_L - 1,
                    )
                    self.mixer = ParallelHyenaConvJumpy(
                        global_config, init_method, layer_number
                    )
                elif self.attention_type == "hyena_outer":
                    from savanna.model.hyena import ParallelHyenaConv2

                    grouped_proj_size = (
                        global_config.hidden_size // global_config.proj_groups
                    )
                    self.hyena_proj_conv = nn.Conv1d(
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        self.hidden_size_per_partition + 2 * grouped_proj_size,
                        global_config.short_conv_L,
                        groups=self.hidden_size_per_partition + 2 * grouped_proj_size,
                        padding=global_config.short_conv_L - 1,
                    )
                    self.mixer = ParallelHyenaConv2(
                        global_config, init_method, layer_number
                    )
                    print(self.mixer)
            elif self.use_flash_attention:
                from savanna.model.flash_attention import (
                    flash_attn_unpadded_qkvpacked_func_cuda,
                    flash_attn_unpadded_kvpacked_func_cuda,
                    flash_attn_unpadded_unpacked_func_triton,
                )

                self.flash_triton_fn = flash_attn_unpadded_unpacked_func_triton
                self.flash_qkv_fn = flash_attn_unpadded_qkvpacked_func_cuda
                self.flash_kv_fn = flash_attn_unpadded_kvpacked_func_cuda

                if self.pos_emb == "alibi":
                    raise ValueError(
                        "Flash attention is currently not compatible with AliBi positional embeddings. Use sinuisoidal, learned, or rotary embeddings instead."
                    )

            elif self.use_flash_attention_v2:
                from flash_attn import (
                    flash_attn_qkvpacked_func,
                    flash_attn_kvpacked_func,
                    flash_attn_func,
                )
                from savanna.model.flash_attention import FlashSelfAttention

                self.attn = FlashSelfAttention(causal=True)

            else:
                self.scale_mask_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=self.fp16,
                    input_in_bf16=self.bf16,
                    fusion_type=get_fusion_type(global_config),
                    mask_func=self.attention_mask_func,
                    softmax_in_fp32=self.attention_softmax_in_fp32,
                    scale=coeff,
                )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = global_config.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            global_config=global_config,
            input_size=global_config.hidden_size,
            output_size=global_config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

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

        if self.pos_emb != "alibi":
            # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
            key_layer = key_layer.transpose(0, 1).reshape(
                output_size[0] * output_size[3], 1, output_size[1], -1
            )
            value_layer = value_layer.transpose(0, 1).reshape(
                output_size[0] * output_size[3], 1, output_size[1], -1
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

            if False:
                query_layer = query_layer.transpose(0, 1).reshape(
                    output_size[0] * output_size[2], 1, output_size[1], -1
                )
                qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

                if self.attention_type == "flash":
                    output = self.flash_qkv_fn(
                        qkv,
                        cu_seqlens_q,
                        max_seqlen_q,
                        self.dropout_p if self.training else 0.0,
                        softmax_scale=None,
                        causal=True,
                    )
                elif self.attention == "flash_v2":
                    output = self.attn(qkv, True, cu_seqlens_q, max_seqlen_q)

            else:
                # [sq, b, np, hn] -> [b * sq, 1, np, hn]
                query_layer = query_layer.transpose(0, 1).reshape(
                    output_size[0] * output_size[2], 1, output_size[1], -1
                )

                # Combined q/k/v into [b * s, 3, np, hn].
                qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

                if self.attention_type == "flash":
                    output = self.flash_qkv_fn(
                        qkv,
                        cu_seqlens_q,
                        max_seqlen_q,
                        self.dropout_p if self.training else 0.0,
                        softmax_scale=None,
                        causal=True,
                    )

                elif self.attention_type == "flash_v2":
                    output = self.attn(
                        qkv,
                        True,
                        cu_seqlens_q,
                        max_seqlen_q,
                    )

            # [b * sq, np, hn] -> [b, sq, np, hn]
            matmul_result = output.view(
                output_size[0], output_size[2], output.shape[1], output.shape[2]
            )
            # [b, sq, np, hn] -> [b, np, sq, hn]
            matmul_result = matmul_result.transpose(1, 2)

        else:
            # [sq, b, np, hn] -> [b, sq, np, hn]
            sq = query_layer.size(0)
            b = query_layer.size(1)
            sk = key_layer.size(0)

            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
            value_layer = value_layer.transpose(0, 1)

            bias = self.alibi_embed.bias(sq, sk, query_layer.device, query_layer.dtype)
            bias = bias.unsqueeze(0).tile((b, 1, 1, 1))

            matmul_result = self.flash_triton_fn(
                query_layer, key_layer, value_layer, bias=bias, causal=True
            )
            matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

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
                self.hidden_size_per_attention_head
                * self.num_attention_heads_per_partition :,
            ]
            grouped_projs = torch.repeat_interleave(grouped_projs, proj_groups, dim=-1)
            x = torch.cat(
                [
                    x[
                        ...,
                        : self.hidden_size_per_attention_head
                        * self.num_attention_heads_per_partition,
                    ],
                    grouped_projs,
                ],
                dim=-1,
            )

            # hidden_size_grouped_proj = (
            #     self.hidden_size_per_attention_head // proj_groups
            # )
            # new_tensor_shape = x.size()[:-1] + (
            #     self.num_attention_heads_per_partition,
            #     self.hidden_size_per_attention_head + 2 * hidden_size_grouped_proj,
            # )
            # x = x.view(*new_tensor_shape)

            # # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            # x1, x2, value = torch.split(
            #     x,
            #     [
            #         self.hidden_size_per_attention_head,
            #         hidden_size_grouped_proj,
            #         hidden_size_grouped_proj,
            #     ],
            #     dim=-1,
            # )

            new_tensor_shape = x.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            x = x.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (x1, x2, value) = mpu.split_tensor_along_last_dim(x, 3)

        return x1, x2, value

    def forward(self, u, attention_mask, layer_past=None):
        """
        Applies projections and mixing to a sequence u: L x B x D
        """

        # L x B x D --> : L x B x (NP * 3 * DP)
        if self.attention_type == "hyena_v2":
            L = u.size(0)
            # mixed_x_layer, _ = self.query_key_value(hidden_states)
            mixed_x_layer = u
            mixed_x_layer = self.hyena_proj_conv(mixed_x_layer.permute(1, 2, 0))[
                ..., :L
            ].permute(2, 0, 1)

            # L, B, (NP * 3 * DP) --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 2 * hn] --> 3 [sq, b, np, hn]
            (x1, x2, v) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

            context_layer = self.mixer(x1, x2, v)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.hidden_size_per_partition,
            )
            context_layer = context_layer.view(*new_context_layer_shape)

            # =================
            # Output. [sq, b, h]
            # =================
            output, bias = self.dense(context_layer)

            if self.use_cache:
                output = [output, present]

            return output, bias

        else:
            mixed_x_layer, _ = self.query_key_value(u)
            if self.use_hyena:
                L = mixed_x_layer.size(0)
                mixed_x_layer = self.hyena_proj_conv(mixed_x_layer.permute(1, 2, 0))[
                    ..., :L
                ].permute(2, 0, 1)
            if self.grouped_attention and self.use_hyena:
                query, key_layer, value_layer = self.split_projections(mixed_x_layer, 1)
                # query_shape = mixed_x_layer.size()[:2] + (self.hidden_size,)
                # kv_shape = mixed_x_layer.size()[:2] + (self.grouped_proj_size,)
                # query = mixed_x_layer[..., :self.hidden_size]
                # key_layer, value_layer = torch.split(mixed_x_layer[..., self.hidden_size:], self.grouped_proj_size, dim=-1)
            else:
                query, key_layer, value_layer = self.split_projections(
                    mixed_x_layer, self.proj_groups
                )

            if exists(self.rotary_emb):
                if exists(self.rotary_ndims):
                    # partial rotary
                    query_rot, query_pass = (
                        query[..., : self.rotary_ndims],
                        query[..., self.rotary_ndims :],
                    )
                    key_rot, key_pass = (
                        key_layer[..., : self.rotary_ndims],
                        key_layer[..., self.rotary_ndims :],
                    )
                else:
                    # full rotary
                    query_rot, key_rot = query, key_layer
                apply_rotary_fn = (
                    apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
                )

                seq_len = key_layer.shape[0]
                offset = 0
                if exists(layer_past) and layer_past.numel() > 0:
                    offset = layer_past[0].shape[0]
                    seq_len += offset
                cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
                query, key_layer = apply_rotary_fn(
                    query_rot, key_rot, cos, sin, offset=offset
                )

                if exists(self.rotary_ndims):
                    query = torch.cat((query, query_pass), dim=-1)
                    key_layer = torch.cat((key_layer, key_pass), dim=-1)

            # ==================================
            # Cache key and value for inference
            # ==================================

            if exists(layer_past) and layer_past.numel() > 0:
                past_key, past_value = layer_past
                key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
                value_layer = torch.cat(
                    (past_value.type_as(value_layer), value_layer), dim=0
                )

            if self.use_cache:
                present = torch.stack((key_layer, value_layer))

            if self.use_h3 or self.use_badger or self.use_hyena:
                context_layer = self.mixer(query, key_layer, value_layer)

            elif self.use_flash_attention or self.use_flash_attention_v2:
                context_layer = self.flash_attention(query, key_layer, value_layer)

            elif not self.sparse:
                context_layer = self.attention(
                    query, key_layer, value_layer, layer_past, attention_mask
                )
            else:
                context_layer = self.sparse_attention(
                    query, key_layer, value_layer, attention_mask
                )

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.hidden_size_per_partition,
            )
            context_layer = context_layer.view(*new_context_layer_shape)

            # =================
            # Output. [sq, b, h]
            # =================
            output, bias = self.dense(context_layer)

            if self.use_cache:
                output = [output, present]

            return output, bias


# class ParallelTransformerLayer(nn.Module):
#     """A single transformer layer.

#     Transformer layer takes input with size [b, s, h] and returns an
#     output of the same size.
#     """

#     def __init__(
#         self,
#         global_config,
#         attention_mask_func,
#         init_method,
#         output_layer_init_method,
#         layer_number,
#         rpe=None,
#         rotary=False,
#         use_cache=False,
#     ):
#         super().__init__()

#         self.layer_number = layer_number
#         self.dtype = get_dtype_from_string(global_config.precision)
#         self.log_attn_norms = global_config.log_attn_norms
#         self.global_config = global_config

#         norm, eps = get_norm(global_config)
#         self.prenorm, self.postnorm = global_config.prenorm, global_config.postnorm
#         if global_config.prenorm:
#             self.input_layernorm = norm(global_config.hidden_size, eps=eps)

#         self.attention_type = global_config.attention_config[layer_number]
#         # if global_config.postnorm and self.attention_type != "hyena":
#         self.post_attention_layernorm = norm(global_config.hidden_size, eps=eps)

#         self.use_cache = use_cache

#         self.hidden_dropout = global_config.hidden_dropout
#         self.bias_dropout_fusion = global_config.bias_dropout_fusion
#         self.gpt_j_residual = global_config.gpt_j_residual
#         self.gpt_j_tied = global_config.gpt_j_tied
#         self.mlp_type = global_config.mlp_type

#         if self.gpt_j_residual:
#             self.reduce = mpu.mappings.reduce_from_model_parallel_region

#         # Self attention.
#         self.attention = ParallelSelfAttention(
#             global_config=global_config,
#             attention_mask_func=attention_mask_func,
#             init_method=init_method,
#             output_layer_init_method=output_layer_init_method,
#             layer_number=layer_number,
#             rpe=rpe,
#             use_cache=self.use_cache,
#             rotary=rotary,
#             parallel_output=self.gpt_j_residual,
#         )

#         # MLP
#         if global_config.all_config["identity_mlp"]:
#             self.mlp = nn.Identity()
#         else:
#             if global_config.mlp_type == "regular":
#                 self.mlp = ParallelMLP(
#                     global_config=global_config,
#                     init_method=init_method,
#                     output_layer_init_method=output_layer_init_method,
#                     parallel_output=self.gpt_j_residual,
#                 )
#             else:
#                 self.mlp = ParallelGatedMLP(
#                     global_config=global_config,
#                     init_method=init_method,
#                     output_layer_init_method=output_layer_init_method,
#                     parallel_output=self.gpt_j_residual,
#                 )

#         self.layer_past = None  # used to cache k/v pairs in inference

#     def _get_bias_dropout(self):
#         if self.bias_dropout_fusion:
#             fn = (
#                 bias_dropout_add_fused_train
#                 if self.training
#                 else bias_dropout_add_fused_inference
#             )
#         else:
#             fn = get_bias_dropout_add(self.training)
#         return fn

#     def forward(self, x, attention_mask, layer_past=None):
#         if self.dtype != x.dtype:
#             x = x.type(self.dtype)

#         layer_past = layer_past if layer_past is not None else self.layer_past
#         bias_dropout_fn = self._get_bias_dropout()

#         # x: [b, s, h]
#         if self.gpt_j_residual:
#             # pseudocode:
#             # x = x + attn(ln(x)) + mlp(ln(x))
#             # this means we can avoid doing the allreduce in the attn / mlp outputs
#             # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
#             # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
#             # we preserve the functionality for backwards compatibility

#             residual = x
#             # applies the correct normalization depending on if the norms are tied
#             if self.gpt_j_tied:
#                 x = self.input_layernorm(x)
#                 x1, x2 = x, x
#             else:
#                 x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

#             # attention operator
#             attention_output, attention_bias = self.attention(
#                 x1, attention_mask, layer_past=layer_past
#             )
#             if self.use_cache:
#                 attention_output, presents = attention_output
#                 self.layer_past = presents

#             with torch.enable_grad():
#                 attention_output = bias_dropout_fn(
#                     attention_output,
#                     bias=attention_bias.expand_as(attention_output),
#                     residual=None,
#                     prob=self.hidden_dropout,
#                 )

#             # mlp operator
#             # check if identity
#             if isinstance(self.mlp, nn.Identity):
#                 output = attention_output
#             else:
#                 mlp_output, mlp_bias = self.mlp(x2)
#                 with torch.enable_grad():
#                     output = bias_dropout_fn(
#                         mlp_output,
#                         bias=mlp_bias.expand_as(mlp_output),
#                         residual=attention_output,
#                         prob=self.hidden_dropout,
#                     )

#             # output = (x + attn(ln(x)) + mlp(ln(x))
#             output = residual + self.reduce(output)
#         else:
#             # pseudocode:
#             # x = x + attn(ln1(x))
#             # x = x + mlp(ln2(x))

#             residual = x

#             # x = x + attn(ln1(x))
#             # attention_output, attention_bias = self.attention(
#             #     self.input_layernorm(x), attention_mask, layer_past=layer_past
#             # )
#             if self.prenorm:
#                 if torch.distributed.get_rank() == 0 and self.log_attn_norms:
#                     with torch.no_grad():
#                         log_norm(
#                             x.norm(2, dim=-1).max(1).values.mean(0),
#                             f"prenorm_xnorm_{self.layer_number}",
#                             self.global_config.iteration,
#                         )

#                 x = self.input_layernorm(x)
#                 # [MP] Some load_checkpoint configurations load
#                 # rmsnorm parameters as float32 even with bfloat16 active in config
#                 if x.dtype != self.dtype:
#                     x = x.to(self.dtype)

#                 if torch.distributed.get_rank() == 0 and self.log_attn_norms:
#                     with torch.no_grad():
#                         log_norm(
#                             x.norm(2, dim=-1).max(1).values.mean(0),
#                             f"post_prenorm_xnorm_{self.layer_number}",
#                             self.global_config.iteration,
#                         )

#             attention_output, attention_bias = self.attention(
#                 x, attention_mask, layer_past=layer_past
#             )

#             if torch.distributed.get_rank() == 0 and self.log_attn_norms:
#                 with torch.no_grad():
#                     log_norm(
#                         attention_output.norm(2, dim=-1).max(1).values.mean(0),
#                         f"post_attn_norm_{self.layer_number}",
#                         self.global_config.iteration,
#                     )
#                     log_norm(
#                         attention_bias.expand_as(residual)
#                         .norm(2, dim=-1)
#                         .max(1)
#                         .values.mean(0),
#                         f"post_attn_residual_bias_borm_{self.layer_number}",
#                         self.global_config.iteration,
#                     )

#             if self.use_cache:
#                 attention_output, presents = attention_output
#                 self.layer_past = presents
#             with torch.enable_grad():
#                 attention_output = bias_dropout_fn(
#                     attention_output,
#                     bias=attention_bias.expand_as(residual),
#                     residual=residual,
#                     prob=self.hidden_dropout,
#                 )

#             if torch.distributed.get_rank() == 0 and self.log_attn_norms:
#                 with torch.no_grad():
#                     log_norm(
#                         attention_output.norm(2, dim=-1).max(1).values.mean(0),
#                         f"post_attn_residual_norm_{self.layer_number}",
#                         self.global_config.iteration,
#                     )

#             if isinstance(self.mlp, nn.Identity):
#                 output = attention_output

#             else:
#                 # output = x + mlp(ln2(x))
#                 mlp_output, mlp_bias = self.mlp(attention_output)

#                 if torch.distributed.get_rank() == 0 and self.log_attn_norms:
#                     with torch.no_grad():
#                         log_norm(
#                             mlp_output.norm(2, dim=-1).max(1).values.mean(0),
#                             f"post_mlp_norm_{self.layer_number}",
#                             self.global_config.iteration,
#                         )

#                 if (
#                     self.postnorm
#                 ):  # and self.attention_type != "hyena", skip applying this in some checkpoints!
#                     mlp_output = self.post_attention_layernorm(mlp_output)

#                 if torch.distributed.get_rank() == 0 and self.log_attn_norms:
#                     with torch.no_grad():
#                         log_norm(
#                             mlp_output.norm(2, dim=-1).max(1).values.mean(0),
#                             f"post_postnorm_norm_{self.layer_number}",
#                             self.global_config.iteration,
#                         )

#                 if self.mlp_type == "llama":
#                     output = attention_output + mlp_output

#                 else:
#                     with torch.enable_grad():
#                         output = bias_dropout_fn(
#                             mlp_output,
#                             bias=mlp_bias.expand_as(mlp_output),
#                             residual=attention_output,
#                             prob=self.hidden_dropout,
#                         )

#         return output


class ParallelTransformerLayer(nn.Module):
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

        self.attention_type = global_config.attention_config[layer_number]
        # if global_config.postnorm and self.attention_type != "hyena":
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
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        # Self attention.
        self.attention = ParallelSelfAttention(
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
            if global_config.mlp_type == "regular" or (
                self.safe and self.attention_type == "hyena"
            ):
                self.mlp = ParallelMLP(
                    global_config=global_config,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                )
            elif global_config.mlp_type == "doublegate_llama":
                self.mlp = DoubleGateParallelGatedMLP(
                    global_config=global_config,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                )
            else:
                self.mlp = ParallelGatedMLP(
                    global_config=global_config,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                )

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()

        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied:
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1, attention_mask, layer_past=layer_past
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # mlp operator
            # check if identity
            if isinstance(self.mlp, nn.Identity):
                output = attention_output
            else:
                mlp_output, mlp_bias = self.mlp(x2)
                with torch.enable_grad():
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(mlp_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)

        # main mixer layer logic
        else:
            residual = x

            if self.prenorm:
                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            x.norm(2, dim=-1).max(1).values.mean(0),
                            f"prenorm_xnorm_{self.layer_number}",
                            self.global_config.iteration,
                        )

                x = self.input_layernorm(x)

                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            x.norm(2, dim=-1).max(1).values.mean(0),
                            f"post_prenorm_xnorm_{self.layer_number}",
                            self.global_config.iteration,
                        )

            attention_output, attention_bias = self.attention(
                x, attention_mask, layer_past=layer_past
            )

            if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                with torch.no_grad():
                    log_norm(
                        attention_output.norm(2, dim=-1).max(1).values.mean(0),
                        f"post_attn_norm_{self.layer_number}",
                        self.global_config.iteration,
                    )
                    log_norm(
                        attention_bias.expand_as(residual)
                        .norm(2, dim=-1)
                        .max(1)
                        .values.mean(0),
                        f"post_attn_residual_bias_borm_{self.layer_number}",
                        self.global_config.iteration,
                    )

            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual),
                    residual=residual,
                    prob=self.hidden_dropout,
                )

            if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                with torch.no_grad():
                    log_norm(
                        attention_output.norm(2, dim=-1).max(1).values.mean(0),
                        f"post_attn_residual_norm_{self.layer_number}",
                        self.global_config.iteration,
                    )

            if isinstance(self.mlp, nn.Identity):
                output = attention_output

            # main mlp logic
            else:
                # output = x + mlp(ln2(x))
                # option for pre-mlp norm
                if self.pre_mlp_norm:
                    output = self.pre_mlp_layernorm(attention_output)
                else:
                    output = attention_output

                mlp_output, mlp_bias = self.mlp(output)

                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            mlp_output.norm(2, dim=-1).max(1).values.mean(0),
                            f"post_mlp_norm_{self.layer_number}",
                            self.global_config.iteration,
                        )

                # assert not (self.pre_mlp_norm and self.postnorm), "can't do both pre mlp and post norm!"

                if (
                    self.postnorm  # this is after attention, but before residual
                ):  # and self.attention_type != "hyena", skip applying this in some checkpoints!
                    mlp_output = self.post_attention_layernorm(mlp_output)

                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            mlp_output.norm(2, dim=-1).max(1).values.mean(0),
                            f"post_postnorm_norm_{self.layer_number}",
                            self.global_config.iteration,
                        )

                if self.mlp_type == "llama" or self.mlp_type == "doublegate_llama":
                    output = (
                        attention_output + mlp_output
                    )  # attention_output is the residual basically

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
                            residual=attention_output,
                            prob=self.hidden_dropout,
                        )
        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        return super().forward(hidden_states, attention_mask), attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, bias = super().forward(hidden_state)

        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel
    return mpu.gather_from_model_parallel_region(logits_parallel)
