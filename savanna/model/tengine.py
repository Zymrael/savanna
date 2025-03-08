from savanna.mpu import get_model_parallel_group, get_cuda_rng_tracker
from savanna.dtype import get_dtype_from_string

from importlib.metadata import version
from typing import Callable

import torch
from pkg_resources import packaging

try:
    from transformer_engine.pytorch import LayerNormLinear, Linear, LayerNorm
    from transformer_engine.common.recipe import Format, DelayedScaling
except:
    print("WARNING: transformer_engine not installed. Using default recipe.")


def set_format_recipe(global_config):
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_format, fp8_recipe


def _get_extra_te_kwargs(config):
    extra_transformer_engine_kwargs = {}
    from importlib.metadata import version

    from pkg_resources import packaging

    te_version = packaging.version.Version(version("transformer-engine"))
    if te_version >= packaging.version.Version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = "cpu"
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


class TENorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        normalization="LayerNorm",
        **kwargs
    ):
        if normalization == "LayerNorm":
            instance = LayerNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=sequence_parallel,
                **_get_extra_te_kwargs(config),
            )
        elif normalization == "RMSNorm":
            assert hasattr(te, "RMSNorm"), "Transformer-Engine >= v0.11 required to use this feature"
            instance = RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=sequence_parallel,
                **_get_extra_te_kwargs(config),
            )
        else:
            raise Exception("Only LayerNorm and RMSNorm are curently supported")

        return instance


class TELinear(Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config,
        parallel_mode: str,
        init_method: Callable,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        **kwargs
    ):
        self.config = config
        # Parameters are initialized at higher precision even if fp8
        # is used
        params_dtype = get_dtype_from_string(config.params_dtype)

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_model_parallel_group(check_initialized=False),
            tp_size=self.config.model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=init_method,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            bias=bias,
            return_bias=self.te_return_bias,
            **_get_extra_te_kwargs(config),
            **kwargs,
        )

    def forward(self, x):
        out = super().forward(x)

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class TELayerNormColumnParallelLinear(LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        **kwargs
    ):
        self.config = config
        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.11.0"):
            kwargs["normalization"] = self.config.normalization

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=init_method,
            params_dtype=self.config.params_dtype,
            parallel_mode="column",
            return_bias=self.te_return_bias,
            **_get_extra_te_kwargs(config),
            **kwargs,
        )

    def forward(self, x):
        out = super().forward(x)

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(self, input_size: int, output_size: int, config, **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="column",
            **kwargs,
        )


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(self, input_size: int, output_size: int, config, **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="row",
            **kwargs,
        )


# class TEDotProductAttention(DotProductAttention):
#     """
#     Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
#     has "flash attention" enabled.

#     Note that if Megatron's parallel_state has not been initialized
#     yet, the tp_group passed to TE will be None and must be set later
#     via set_tensor_parallel_group().
#     """

#     def __init__(
#         self,
#         config,
#         layer_number: int = 1,
#         attn_mask_type: AttnMaskType = AttnMaskType.padding,
#         **kwargs
#     ):
#         self.config = config
#         super().__init__(
#             num_attention_heads=self.config.num_attention_heads,
#             kv_channels=self.config.kv_channels,
#             attention_dropout=self.config.attention_dropout,
#             layer_number=layer_number,
#             attn_mask_type=attn_mask_type.name,
#             sequence_parallel=self.config.sequence_parallel,
#             tp_size=self.config.tensor_model_parallel_size,
#             get_rng_state_tracker=get_cuda_rng_tracker,
#             tp_group=get_model_parallel_group(check_initialized=False),
#             **kwargs,
#         )


# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# from dataclasses import dataclass
# from typing import Callable

# import torch
# import torch.nn.functional as F

# from megatron.core import ModelParallelConfig
# from megatron.core.utils import init_method_normal, scaled_init_method_normal


# @dataclass
# class TransformerConfig(ModelParallelConfig):
#     """Configuration object for megatron-core transformers.

#         Attributes:

#         # model architecture
#         num_layers (int): Number of transformer layers in a transformer block.
#         hidden_size (int): Transformer hidden size.
#         ffn_hidden_size (int): Transformer Feed-Forward Network hidden size.
#                                 This is set to 4*hidden_size if not provided. Defaults to None.')
#         num_attention_heads (int): Number of transformer attention heads.
#         kv_channels (int): Projection weights dimension in multi-head attention.
#                             This is set to hidden_size // num_attention_heads if not provided.
#                             Defaults to None.
#         num_query_groups (int): Number of query groups for group query attention. If None, normal attention is used.

#         hidden_dropout (float): Dropout probability for transformer hidden state. Defaults to 0.1.
#         attention_dropout (float): Post attention dropout probability. Defaults to 0.1.
#         fp32_residual_connection (bool): If true, move residual connections to fp32.
#         apply_residual_connection_post_layernorm (bool): If true, uses the original BERT residule connection ordering.
#                                                          Defaults to False.
#         layernorm_epsilon (float): Layernorm epsilon. Defaults to 1e-5.

#         layernorm_zero_centered_gamma (bool): if set to 'True', the LayerNorm is adjusted to center the gamma values
#                                               around 0. This improves numerical stability. Defaults to False.

#         add_bias_linear (bool): Include a bias term in all linear layers (QKV projections, after core attention, and two
#                                 in MLP layer). Default is True.

#         gated_linear_unit (bool): Use a gated linear unit for the first linear layer in the MLP. Defaults to False.

#         activation_func (Callable): Activation function to use for the non-linearity in the MLP. Defaults to F.gelu.

#         # initialization
#         init_method (Callable): Method to initialize weights. Note that bias is always set to
#                                 zero. Should be a function that takes a single Tensor and
#                                 initializes it. Defaults to
#                                 megatron.core.utils.init_method_normal(init_method_std) which is
#                                 torch.nn.init.normal_ with mean=0.0 and std=init_method_Std.

#         output_layer_init_method (Callable): Method to initialize weights of the output layer of
#                                              both attention and MLP blocks. Defaults to
#                                              megatron.core.utils.scaled_init_method_normal(init_method_std)
#                                              which is torch.nn.init.normal_ with mean=0.0 and
#                                              std=init_method_std / math.sqrt(2.0 * num_layers).

#         init_method_std (float): Standard deviation of the zero mean normal for the default
#                                  initialization method, not used if init_method and
#                                  output_layer_init_method are provided. Defaults to 0.02.

#         # mixed-precision
#         apply_query_key_layer_scaling (bool): If true, scale Q * K^T by 1 / layer-number. Defaults to True.
#         attention_softmax_in_fp32 (bool): If true, run attention masking and softmax in fp32.
#                                           This should be true if apply_query_key_layer_scaling is true.

#         # fusion
#         bias_gelu_fustion (bool): If true, fuses bias and gelu. Defaults to False.
#         masked_softmax_fusion (bool): If true, uses softmax fusion.
#         persist_layer_norm (bool): If true, uses the persistent fused layer norm kernel.
#                                    This kernel only supports a fixed set of hidden sizes.
#                                    Defaults to False.
#         bias_dropout_fusion (bool): If true, uses bias dropout fusion.

#         # activation recomputation

#         recompute_granularity (str): megatron-core supports 'selective' activation checkpointing where only the memory
#                                      intensive part of attention is checkpointed.  These memory intensive activations
#                                      are also less compute intensive which makes activation checkpointing more efficient
#                                      for LLMs (20B+).  See Reducing Activation Recomputation in Large Transformer
#                                      Models: https://arxiv.org/abs/2205.05198 for more details.  'full' will checkpoint
#                                      the entire transformer layer.  Must be 'selective' or 'full'. Defaults to None.

#         recompute_method (str): uniform will uniformly divide the total number of transformer layers in a transformer
#                                 block and recompute the input activation of each divided chunk at the specified
#                                 granularity.  block will recompute the input activations for only a set number of
#                                 transformer layers per pipeline stage.  The rest of the layers in the pipeline stage
#                                 will not have any activations recomputed.  Must be 'uniform' or 'block'. Defaults to
#                                 None.

#         recompute_num_layers (int): When recompute_method is uniform, recompute_num_layers is the number of transformer
#                                     layers in each uniformly divided recompute unit.  When recompute_method is block,
#                                     recompute_num_layers is the number of transformer layers to recompute within each
#                                     pipeline stage.  Defaults to None.

#         distribute_saved_activations (bool): If true, distribute recomputed activations across the model parallel
#                                              group. Defaults to None.

#         # fp8 related (via Transformer Engine). For detailed info, refer the the Transformer Engine docs at
#         # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html

#         fp8 (str): If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined choices: (1) 'e4m3'
#                    uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8 activation and weight tensors and
#                    e5m2 for all FP8 output activation gradient tensors. Defaults to None.

#         fp8_margin (int): Margin for the scaling factor computation.

#         fp8_interval (int): Controls how often the scaling factor is recomputed.

#         fp8_amax_history_len (int): The length of the amax history window used for scaling factor computation.

#         fp8_amax_compute_algo (str): Algorithm used for choosing the `amax` value for the scaling factor computation.
#                                      There are 2 predefined choices: `max` chooses the largest `amax` in the history
#                                      window, while `most_recent` always chooses the most recently seen value.

#         fp8_wgrad (bool): When set to False, override FP8 config options and do the wgrad computation in higher precision.
#                           Defaults to True.

#         # Experimental
#         normalization (str): Swtich b/w `LayerNorm` and `RMSNorm` as normalization layers. For now, these are primarily
#                              used by Transformer-Engine's layers like `LayerNormLinear`. Default value is `LayerNorm`.


#     """

#     # model architecture
#     num_layers: int = 0
#     hidden_size: int = 0
#     num_attention_heads: int = 0
#     num_query_groups: int = None

#     ffn_hidden_size: int = None
#     kv_channels: int = None
#     hidden_dropout: float = 0.1
#     attention_dropout: float = 0.1
#     fp32_residual_connection: bool = False
#     # @jcasper should we keep this option?
#     apply_residual_connection_post_layernorm: bool = False
#     layernorm_epsilon: float = 1e-5
#     layernorm_zero_centered_gamma: bool = False
#     add_bias_linear: bool = True
#     gated_linear_unit: bool = False
#     activation_func: Callable = F.gelu

#     # initialization
#     init_method: Callable = None
#     output_layer_init_method: Callable = None
#     init_method_std: float = 0.02

#     # mixed-precision
#     apply_query_key_layer_scaling: bool = True
#     attention_softmax_in_fp32: bool = True

#     # communication

#     # fusion
#     bias_gelu_fusion: bool = False  # TODO: this should be bias_activation_fusion ?
#     masked_softmax_fusion: bool = False
#     persist_layer_norm: bool = False
#     bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?

#     # activation recomputation
#     recompute_granularity: str = None
#     recompute_method: str = None
#     recompute_num_layers: int = None
#     distribute_saved_activations: bool = None

#     # fp8 related
#     fp8: str = None
#     fp8_margin: int = 0
#     fp8_interval: int = 1
#     fp8_amax_history_len: int = 1
#     fp8_amax_compute_algo: str = "most_recent"
#     fp8_wgrad: bool = True

#     # experimental section (TODO: move to apt. section above once stable)
#     normalization: bool = "LayerNorm"  # alt value supported by TE: "RMSNorm"

#     def __post_init__(self):
#         """ Python dataclass method that is used to modify attributes after initialization.
#             See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
#         """
#         super().__post_init__()
#         if self.fp16 and self.bf16:
#             raise ValueError(
#                 f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
#             )

#         if self.num_attention_heads % self.tensor_model_parallel_size != 0:
#             raise ValueError(
#                 f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
#                 f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
#             )

#         if self.ffn_hidden_size is None:
#             self.ffn_hidden_size = 4 * self.hidden_size

#         if self.kv_channels is None:
#             self.kv_channels = self.hidden_size // self.num_attention_heads

#         if self.num_query_groups is None:
#             self.num_query_groups = self.num_attention_heads

#         if self.num_query_groups % self.tensor_model_parallel_size != 0:
#             raise ValueError(
#                 f"num_query_groups ({self.num_query_groups}) must be a multiple of "
#                 f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
#             )

#         if self.apply_query_key_layer_scaling:
#             self.attention_softmax_in_fp32 = True

#         if self.recompute_granularity is not None:
#             if not self.recompute_granularity in ['full', 'selective']:
#                 raise ValueError(
#                     f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
#                 )

#             if self.recompute_method is not None:
#                 if not self.recompute_method in ['block', 'uniform']:
#                     raise ValueError(
#                         f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
#                     )
#             elif self.recompute_granularity != 'selective':
#                 raise ValueError(
#                     f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
#                 )

#             if self.recompute_num_layers is None:
#                 raise ValueError(
#                     f'When using recompute_granularity: {self.recompute_granularity} so recompute_num_layers must be between '
#                     f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
#                 )

#             if self.distribute_saved_activations and self.sequence_parallel:
#                 raise ValueError(
#                     f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel}'
#                 )

#             if self.virtual_pipeline_model_parallel_size is not None:
#                 if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
#                     raise ValueError(
#                         f'num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
#                     )

#         if self.apply_query_key_layer_scaling:
#             self.attention_softmax_in_fp32 = True

#         if self.bias_gelu_fusion:
#             if not self.add_bias_linear:
#                 raise ValueError(
#                     "When bias_gelu_fusion is True, add_bias_linear must also be True."
#                 )

#             if self.activation_func != F.gelu:
#                 raise ValueError(f'When bias_gelu_fusion is True, activation_func must be F.gelu.')

#         if self.init_method is None:
#             self.init_method = init_method_normal(self.init_method_std)

#         if self.output_layer_init_method is None:
#             self.output_layer_init_method = scaled_init_method_normal(
#                 self.init_method_std, self.num_layers
#             )
