import math
from dataclasses import asdict, dataclass, fields

import torch

from savanna.arguments import GlobalConfig

"""
MFU = Achieved FLOPs per s / Device FLOPs per s

Achieved FLOPs per s = FLOPs_per_batch / iteration_time_per_batch
FLOPS_per_batch = global_batch_size * seqlen * FLOPs_per_token
FLOPS_per_token = embedding_layer_flops + sequence_mixer_flops + num_layers * FFN_flops + logits_flops

_Sequence Mixer_
sequence_mixer_flops = num_attn_layer * attn_flops + num_hyena_layers * hyena_conv_flops
- attn_flops = qkv_proj_flops + qk_logit_flops + attention_over_values_flops + out_proj_flops 
    - qkv_proj_flops = 3 * 2 * bs * seqlen * d_model * d_model
    - qk_logit_flops = 2 * bs * seqlen * seqlen * d_model
    - attention_over_values_flops = qk_logit_flops
    - out_proj_flops = 2 * bs * seqlen * d_model * d_model
- hyena_conv_flops = hyena_conv_proj_flops + hyena_linear_proj_flops + short_conv_flops + medium_conv_flops + long_conv_flops
  -> assumes all convs are depthwise and are implemented as implicit (batched) gemms, where M = seqlen, K = kernel_size, and N = d_model 
  - hyena_conv_proj_flops = num_hyena_layers * 2 * (bs * seqlen) * 3 * d_model * 3, where the last factor of 3 is for qkv
  -> Assumes conv proj uses a kernel_size of 3
  - hyena_linear_proj_flops = num_hyena_layers * 2 * bs * seqlen * d_model * d_model * 3
  -num_hyena_layers = num_short_conv_layers + num_medium_conv_layers + num_long_conv_layers
  - {short_conv_flops, medium_conv_flops, long_conv_flops} = num_{s,m,l}_conv_layers * (2 * bs * seqlen * kernel_size_{s,m,l} * d_model)
_FFN_
- 2 * 3 * bs * seqlen * (8 / 3) * d_model * d_model
    - Factor of 3 from w1, w2, an w3
    - Factor of (8 / 3) from ParallelGLU ffn_dim
     
_Embedding / Logits_
embedding / logits flops = 2 * bs * seqlen * d_model * vocab_size 
-> embedding flops not included

TODO: 
- Fine-grained accounting of activation checkpointing - e.g., selective recompute where only a subset of the layers are recomputed
"""

# ---- Constants ---- #
VOCAB_SIZE = 512
FFN_EXPANSION_FACTOR = 8 / 3
FFN_DIM_MULTIPLE = 64


# ------------------------------------------- Hyena Flops ------------------------------------------- #
@dataclass
class HyenaFlopCounts:
    """
    Flop counts by layer type for Hyena Model
    Pass = fwd pass through the model
    
    """
    # dense flops (except for hyena_conv_proj which is a convolution)
    dense_proj_flops: int
    hyena_conv_proj_flops: int
    out_proj_flops: int
    ffn_flops: int
    # attn flops
    transformer_attn_flops: int
    hyena_conv_flops: int
    # logits flops
    logits_flops: int
    # totals    
    total_dense_linear_flops: int = None
    total_attn_flops: int = None
    total_flops: int = None

    def __post_init__(self):
        total_flops = (
            + self.dense_proj_flops
            + self.hyena_conv_proj_flops
            + self.out_proj_flops
            + self.ffn_flops
            + self.transformer_attn_flops
            + self.hyena_conv_flops
            + self.logits_flops
        )
        if self.total_flops is None:
            self.total_flops = total_flops
        else:
            assert self.total_flops == total_flops   
        
        total_attn_flops = self.transformer_attn_flops + self.hyena_conv_flops
        if self.total_attn_flops is None:
            self.total_attn_flops = total_attn_flops
        else:
            assert self.total_attn_flops == total_attn_flops
      
        total_dense_linear_flops = self.dense_proj_flops + self.out_proj_flops + self.ffn_flops
        if self.total_dense_linear_flops is None:
            self.total_dense_linear_flops = total_dense_linear_flops
        else:
            assert self.total_dense_linear_flops == total_dense_linear_flops
    
    def __str__(self):
        field_strings = []
        for field in fields(self):
            value = getattr(self, field.name)
            # Format the output in scientific notation for the flop fields
            if isinstance(value, int):
                field_strings.append(f"{field.name}: {value:.1e}")
            else:
                field_strings.append(f"{field.name}: {value}")
        return "\n".join(field_strings)

@dataclass
class HyenaFlopsPerIter(HyenaFlopCounts):
    """
    Flop counts per iteration by layer type for Hyena Model
    Iteration = 1 fwd pass + 1 bwd pass
    Assumes bwd pass flops = 2x fwd pass flops
    
    If activation_checkpointing is True, then another fwd pass flops is added.
    """
    activation_checkpointing: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        num_passes = 4 if self.activation_checkpointing else 3
        for f in fields(HyenaFlopCounts):
            setattr(self, f.name, getattr(self, f.name) * num_passes)
            
def add_model_flop_utilization_inputs(global_config):
    """
    Updates global_config with model flops, hardware model flops per iteration, and theoretical flops/s
    
    Used for logging MFU / HFU.
    
    Theoretical flops/s are calculated based on the proportion of model fp8 / bf16 ops.
    Base assumption is that all ops are bf16.  If use_fp8_linears is True, then the theoretical flops/s will
    be blended average of (dense linear flops / total flops) * fp8 flops/s + ((total_flops - dense linear flops ) / total flops) * bf16 flops/s 
     
    Note: flops is floating point operations as opposed to flops per second (flops/s)
    
    TODO:
    - Adjust theoretical flops based on proportion of fp8 / bf16 ops
    """
    
    flop_counts: HyenaFlopsPerIter = get_hyena_flops(global_config=global_config, num_gpus=torch.distributed.get_world_size(), activation_checkpointing=False)
    # Flop counts for MFU
    global_config.model_flops_per_iteration = flop_counts.total_flops
    # Flop counts for HFU
    global_config.hw_model_flops_per_iteration = 4 * int(flop_counts.total_flops / 3) if global_config.checkpoint_activations else flop_counts.total_flops
    
    # Theoretical flops per s for device
    fp16_throughput = get_available_flops(device=torch.device("cuda"), dtype=torch.bfloat16)
    fp8_throughput = get_available_flops(device=torch.device("cuda"), dtype=torch.int8)

    if global_config.use_fp8_linears:
        # Note, int8 throughput == fp8 throughput
        fp16_flops = flop_counts.total_flops - flop_counts.total_dense_linear_flops
        fp8_flops = flop_counts.total_dense_linear_flops
        theoretical_device_throughput = int((fp8_throughput * fp8_flops + fp16_throughput * fp16_flops) / flop_counts.total_flops)
    else:
        theoretical_device_throughput = fp16_throughput
    
    global_config.theoretical_device_throughput = theoretical_device_throughput
    
    # Add these static inputs to wandb.config
    if global_config.use_wandb:
        import wandb
        wandb.config["flop_counts"] = asdict(flop_counts)
        wandb.config["model_flops_per_iteration"] = global_config.model_flops_per_iteration
        wandb.config["hw_model_flops_per_iteration"] = global_config.hw_model_flops_per_iteration
        wandb.config["device_fp16_throughput"] = fp16_throughput
        wandb.config["device_fp8_throughput"] = fp8_throughput
        wandb.config["theoretical_device_throughput"] = global_config.theoretical_device_throughput
def get_input_shapes(global_config, num_gpus):
    """
    Get input shapes for hyena model
    bs, seqlen, d_model
    """
    bs = get_batch_size(global_config, num_gpus)
    seqlen = global_config.seq_length
    d_model = global_config.hidden_size
    return bs, seqlen, d_model


def get_hyena_flops(
    global_config: GlobalConfig,
    num_gpus: int = None,
    activation_checkpointing: bool = False,
) -> HyenaFlopsPerIter:
    """
    Calculate FLOPS per batch for hyena model

    Args:
    - global_config: GlobalConfig
    - num_gpus: int - should be either specified in global_config or passed as an argument (e.g., dist.get_world_size())
    - activation_checkpointing: bool, purpose of including this flag even though activation_checkpointing can be deduced from `global_config`
    is for calculating MFU, where we want to count only flops for only the fwd / bwd passes regardless if how training is actually done.
     
    Assumptions:
    - does not count embedding flops
    - assumes default `vocab_size` == 512
    - num_layers = num_transformer_attn_layers + num_hyena_layers
        - transformer attn layers use flashattention (only checks for `flash` in operator_config)
        - hyena layers = num_short_conv_layers + num_medium_conv_layers + num_long_conv_layers
    - pre-attn projections for hyena layers are comprised of dense projection and hyena conv projection
        - dense projection implemented as qkv projection (3 * 2 * bs * seqlen * d_model * d_model)
        - hyena conv projection is implemented as a short conv with kernel_size = short-conv-L
    - ffn expansion dimension is 8 / 3 and rounded to nearest multiple of 64
    - long convs are implemented as FFTs
    - short and medium convs are implemented as implicit (batched) gemms
    
    TODO:
    - More fine-grained accounting of activation checkpointing -- incorporate factor for selective recompute
    """
    assert (
        num_gpus is not None or global_config.num_gpus is not None
    ), "num_gpus must be specified either in global_config or as an argument"
    if num_gpus is None:
        num_gpus = global_config.num_gpus
    bs, seqlen, d_model = get_input_shapes(global_config, num_gpus)

    num_layers = global_config.num_layers

    # Parse layer types - checks that returned layers sum to global_config.num_layers
    num_transformer_attention_layers, *num_conv_mixer_layers = parse_layer_types(
        global_config
    )
    num_short_conv_layers, num_medium_conv_layers, num_long_conv_layers = (
        num_conv_mixer_layers
    )
    num_hyena_layers = sum(num_conv_mixer_layers)

    vocab_size = getattr(global_config, "vocab_size", VOCAB_SIZE)

    hyena_medium_conv_len = global_config.hyena_medium_conv_len
    hyena_short_conv_len = global_config.hyena_short_conv_len
    hyena_conv_proj_len = global_config.short_conv_L

    # Logits
    logits_flops = int(logits_flops_per_batch(bs, seqlen, d_model, vocab_size))

    # Mixer layers - qkv, out_proj, and ffn common to all mixer types
    # 1. "qkv" projection flops
    dense_proj_flops = int(num_layers * dense_projection_flops_per_layer(
        bs, seqlen, d_model))
    # 2. post-attn projection flops
    out_proj_flops = int(num_layers * attention_out_proj_flops_per_layer(
        bs, seqlen, d_model))
    # 3. ffn flops
    ffn_flops = int(num_layers * ffn_flops_per_layer(
        bs, seqlen, d_model, ffn_expansion_factor=FFN_EXPANSION_FACTOR
    ))

    # transformer attn flops
    transformer_attn_flops = int(
        num_transformer_attention_layers
        * attention_flops_per_layer(bs, seqlen, d_model)
    )

    # hyena flops
    # hyena layers have an additional hyena conv proj
    hyena_proj_flops = int(num_hyena_layers * hyena_conv_proj_flops_per_layer(
        bs, seqlen, d_model, kernel_size=hyena_conv_proj_len
    ))
    # hyena conv flops
    short_conv_flops = num_short_conv_layers * depthwise_conv_flops_per_layer(
        bs, seqlen, d_model, hyena_short_conv_len
    )
    medium_conv_flops = num_medium_conv_layers * depthwise_conv_flops_per_layer(
        bs, seqlen, d_model, hyena_medium_conv_len
    )
    long_conv_flops = num_long_conv_layers * fft_conv_flops_per_layer(
        bs, seqlen, d_model
    )
    hyena_conv_flops = int(
        short_conv_flops + medium_conv_flops + long_conv_flops
    )

    # total mixer flops
    total_attn_flops = transformer_attn_flops + hyena_conv_flops
    total_flops_per_pass = (
        dense_proj_flops + hyena_proj_flops + out_proj_flops + ffn_flops + total_attn_flops + logits_flops
    )

    flop_counts = HyenaFlopsPerIter(
        dense_proj_flops=dense_proj_flops,
        hyena_conv_proj_flops=hyena_proj_flops,
        out_proj_flops=out_proj_flops,
        ffn_flops=ffn_flops,
        transformer_attn_flops=transformer_attn_flops,
        hyena_conv_flops=hyena_conv_flops,
        logits_flops=logits_flops,
        total_attn_flops=total_attn_flops,
        total_flops=total_flops_per_pass,
        activation_checkpointing=activation_checkpointing
    )
    return flop_counts


def get_batch_size(global_config, num_gpus):

    assert (
        global_config.train_batch_size or global_config.train_micro_batch_size_per_gpu
    ), "train_batch_size or train_micro_batch_size_per_gpu must be specified"
    if global_config.train_batch_size is not None:
        bs = global_config.train_batch_size
    else:
        bs = (
            global_config.train_micro_batch_size_per_gpu
            * num_gpus
            * global_config.gradient_accumulation_steps
        )
    return bs


def parse_layer_types(global_config):
    num_transformer_attn_layers = sum(
        "flash" in operator for operator in global_config.operator_config
    )
    num_short_conv_layers = sum(
        "short" in operator for operator in global_config.operator_config
    )
    num_medium_conv_layers = sum(
        "medium" in operator for operator in global_config.operator_config
    )
    num_hyena_layers = sum(
        "hyena" in operator for operator in global_config.operator_config
    )
    num_long_conv_layers = (
        num_hyena_layers - num_short_conv_layers - num_medium_conv_layers
    )
    assert (
        num_transformer_attn_layers
        + num_short_conv_layers
        + num_medium_conv_layers
        + num_long_conv_layers
        == len(global_config.operator_config)
        == global_config.num_layers
    )
    return (
        num_transformer_attn_layers,
        num_short_conv_layers,
        num_medium_conv_layers,
        num_long_conv_layers,
    )


def round_to_nearest_power_of_10(n):
    if n == 0:
        return 0
    # Get the exponent of 10 by taking log10
    exponent = math.floor(math.log10(n))
    # Calculate the nearest power of 10 using rounding
    rounded_number = int(round(n, -exponent))
    return rounded_number


def count_flops(global_config: GlobalConfig):
    """
    Counts total flops per fwd pass

    See note for methodology.
    """
    pass


def dense_projection_flops_per_layer(bs, seqlen, d_model):
    """
    qkv projection flops
    """
    return 2 * 3 * bs * seqlen * d_model * d_model


def hyena_conv_proj_flops_per_layer(bs, seqlen, d_model, kernel_size=3):
    """
    hyena conv proj flops

    """
    return 2 * 3 * bs * seqlen * kernel_size * d_model


def attention_out_proj_flops_per_layer(bs, seqlen, d_model):
    """
    Output project after context layer
    """
    return 2 * bs * seqlen * d_model * d_model


def logits_flops_per_batch(bs, seqlen, d_model, vocab_size):
    """
    Flops for final logits layer

    Assumption:
    - This is a batched GEMM with B = bs, M = seqlen, K = d_model, N = vocab_size
    """
    return 2 * bs * seqlen * d_model * vocab_size


def round_to_next_multiple(n, multiple):
    return multiple * ((n + multiple - 1) // multiple)


def ffn_flops_per_layer(
    bs,
    seqlen,
    d_model,
    ffn_expansion_factor=FFN_EXPANSION_FACTOR,
    multiple_of=FFN_DIM_MULTIPLE,
):
    """
    Flops for FFN layer ~ w2(w1 * w3)

    Assumptions:
    - 3 batched GEMMs
        - w1, w3 - M = seqlen, K = d_model, N = ffn_expansion_factor * d_model
        - w2 - M = seqlen, K = ffn_expansion_factor * d_model, N = d_model
        -> Total ffn flops per layer = 2 * 3 * bs * seqlen * (ffn_expansion_factor * d_model) * d_model
    """
    # https://github.com/Zymrael/savanna/blob/cc7282964ead671241cedf2ecf92cc539fa6dab0/savanna/model/operators/local/base.py#L107
    ffn_dim = int(d_model * ffn_expansion_factor)
    ffn_dim = round_to_next_multiple(ffn_dim, multiple_of)
    ffn_flops_per_layer = 2 * 3 * bs * seqlen * ffn_dim * d_model
    return ffn_flops_per_layer


def depthwise_conv_flops_per_layer(bs, seqlen, d_model, kernel_size):
    """
    Single depthwise convolution flop count

    Assumptions:
    - Convolution is depthwise with input channels = output channels
    - Calculated as an implicit GEMM, with M = seqlen, K = kernel_size, N = output_channels
    """
    return 2 * bs * seqlen * kernel_size * d_model


def fft_conv_flops_per_layer(bs, seqlen, d_model):
    """
    Convolution using FFT
    """
    return bs * 10 * seqlen * math.log2(seqlen) * d_model


# ------------------------------------------- Reference ------------------------------------------- #


def palm_transformer_flops_per_iteration_with_attn(
    num_params: int,
    num_layers: int,
    d_model: int,
    bs: int,
    seqlen: int,
    activation_checkpointing: bool = False,
    include_attn: int = True,
    verbose: bool = False,
):
    """Computes the flops per iteration per formula from PaLM paper (https://arxiv.org/abs/2204.02311) with additional
    attention flops

    Assumptions
    - bwd flops = 2x fwd flops
    - activation checkpointing (defaults to False) - if True, adds additional fwd pass flops to total flops
    - assumes all inputs are packed (no padding)
    - attention calculation assumes full attention matrix is formed disregarding causal masking.  Technically should be half the flops
    (e.g., flash attention v2 does not form full attention matrix) but industry convention seems to keep full flops.
    - does not count logits or elementwise flops
    """

    params_flops_per_token = 2 * num_params
    params_flops_per_batch = bs * seqlen * params_flops_per_token

    attn_flops_per_batch = num_layers * attention_flops_per_layer(bs, seqlen, d_model)

    if verbose:
        # print(f"params_flops_per_token: {params_flops_per_token} {round_to_nearest_power_of_10(params_flops_per_token)}")
        print(
            f"params_flops_per_batch: {round_to_nearest_power_of_10(params_flops_per_batch):.1e}"
        )
        print(
            f"attn_flops_per_batch: {round_to_nearest_power_of_10(attn_flops_per_batch):.1e}"
        )

    fwd_pass_flops = (
        params_flops_per_batch + attn_flops_per_batch
        if include_attn
        else params_flops_per_batch
    )
    bwd_pass_flops = 2 * fwd_pass_flops
    flops_per_iteration = (
        fwd_pass_flops + bwd_pass_flops + fwd_pass_flops
        if activation_checkpointing
        else fwd_pass_flops + bwd_pass_flops
    )

    return flops_per_iteration


def attention_flops_per_layer(bs: int, seqlen: int, d_model: int) -> int:
    """Computes the attention flops per sequence.

    attention flops for single layer = qk_logits_flops + attention_over_values_flops
    Assumptions:
    - disregard softmax flops
    - qk_logits flops = M = seqlen, K = d_model, N = seqlen, attention_over_values_flops = M = seqlen, K = seq_len, N = d_model
    - qkv projection and out projection are accounted for elsewhere (see `flops_per_iteration`)
    """
    return 2 * 2 * (bs * d_model * (seqlen**2))


def megatron_transformer_flops_per_batch(
    bs,
    seqlen,
    num_layers,
    hidden_size,
    vocab_size,
    activation_checkpointing: bool = False,
    verbose=False,
):
    """
    https://github.com/deepakn94/Megatron-DeepSpeed/blob/fd325522da86fe158396544fa9c9a181b9ff1478/megatron/training.py#L681-L687
    """
    dense_flops_per_layer = 24 * bs * seqlen * (hidden_size**2)
    attn_flops_per_layer = 4 * bs * (seqlen**2) * hidden_size
    transformer_flops = num_layers * (dense_flops_per_layer + attn_flops_per_layer)
    logits_flops = 2 * bs * seqlen * hidden_size * vocab_size

    if verbose:
        print(
            f"dense_flops_per_batch: {round_to_nearest_power_of_10(num_layers * dense_flops_per_layer):.1e}"
        )
        print(
            f"attn_flops_per_batch: {round_to_nearest_power_of_10(num_layers * attn_flops_per_layer):.1e}"
        )
        print(
            f"transformer_flops: {round_to_nearest_power_of_10(transformer_flops):.1e}"
        )
        print(f"logits_flops: {round_to_nearest_power_of_10(logits_flops):.1e}")

    activations_factor = 4 if activation_checkpointing else 3
    flops_per_iteration = activations_factor * (transformer_flops + logits_flops)
    # flops_per_iteration = (24 * activations_factor * bs * seqlen * num_layers * (hidden_size**2)) * (1. + (seqlen / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    return flops_per_iteration


_CUDA_FLOPS = {
    # Hopper
    # source: https://resources.nvidia.com/en-us-tensor-core
    "h100 nvl": {
        torch.float64: 67e12,
        torch.float32: 133.8e12,
        "tfloat32": 989.4e12,
        torch.bfloat16: 1978.8e12,
        torch.float16: 1978.8e12,
        torch.int8: 3957.8e12,
    },
    "h100 sxm": {
        torch.float64: 33.5e12,
        torch.float32: 66.9e12,
        "tfloat32": 494.7e12,
        torch.bfloat16: 989.4e12,
        torch.float16: 989.4e12,
        torch.int8: 1978.9e12,
    },
    "h100 pcie": {
        torch.float64: 25.6e12,
        torch.float32: 51.2e12,
        "tfloat32": 378e12,
        torch.bfloat16: 756e12,
        torch.float16: 756e12,
        torch.int8: 1513e12,
    },
    "l4": {
        torch.float32: 30.3e12,
        "tfloat32": 60e12,
        torch.bfloat16: 121e12,
        torch.float16: 121e12,
        torch.int8: 242e12,
        "int4": 484e12,
    },
    "l40": {
        torch.float32: 90.5e12,
        "tfloat32": 90.5e12,
        torch.bfloat16: 181e12,
        torch.float16: 181e12,
        torch.int8: 362e12,
        "int4": 724e12,
    },
    # Ampere
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        torch.float64: 9.7e12,
        torch.float32: 19.5e12,
        "tfloat32": 156e12,
        torch.bfloat16: 312e12,
        torch.float16: 312e12,
        torch.int8: 624e12,
    },
    "a6000": {
        torch.float32: 38.7e12,
        "tfloat32": 77.4e12,
        torch.bfloat16: 38.7e12,
        torch.float16: 38.7e12,
        torch.int8: 309.7e12,
        "int4": 619.3e12,
    },
    "a40": {
        torch.float32: 37.4e12,
        "tfloat32": 74.8e12,
        torch.bfloat16: 37.4e12,
        torch.float16: 37.4e12,
        torch.int8: 299.3e12,
        "int4": 598.7e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {
        torch.float32: 31.2e12,
        "tfloat32": 62.5e12,
        torch.bfloat16: 125e12,
        torch.float16: 125e12,
        torch.int8: 250e12,
        "int4": 500e12,
    }
}

# Credit: https://github.com/Lightning-AI/pytorch-lightning/blob/bc3c9c536dc88bfa9a46f63fbce22b382a86a9cb/src/lightning/fabric/utilities/throughput.py#L521
def get_available_flops(device: torch.device, dtype: torch.dtype):
    """Returns the available theoretical FLOPs.

    This is not achievable flops but an upper bound estimate of possible flops given ideal conditions.
    
    """
    device_name = torch.cuda.get_device_name(device)
    chip = device_name.lower()
    if "h100" in chip:
        if "hbm3" in chip:
            chip = "h100 sxm"
        elif "nvl" in chip:
            chip = "h100 nvl"
        elif "pcie" in chip or "hbm2e" in chip:
            chip = "h100 pcie"
    elif "l4" in chip:
        chip = "l40" if "tesla" in chip else "l4"
    elif "a100" in chip:
        chip = "a100"
    elif "a40" in chip:
        chip = "a40"
    elif "a10g" in chip:
        chip = "a10g"
    else:
        return None
    if chip not in _CUDA_FLOPS:
        return None
    dtype_to_flops = _CUDA_FLOPS[chip]
    if dtype is torch.float32:
        if torch.get_float32_matmul_precision() != "highest":
            dtype = "tfloat32"
    if dtype not in dtype_to_flops:
        return None
    return int(dtype_to_flops[dtype])
    return int(dtype_to_flops[dtype])