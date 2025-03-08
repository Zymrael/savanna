import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)

from savanna.arguments import GlobalConfig
from savanna.mfu import (
    VOCAB_SIZE,
    HyenaFlopsPerIter,
    get_available_flops,
    get_hyena_flops,
    get_input_shapes,
    megatron_transformer_flops_per_batch,
    palm_transformer_flops_per_iteration_with_attn,
    parse_layer_types,
    round_to_nearest_power_of_10,
)

TEST_DIR = Path(__file__).parent.parent
TEST_CONFIGS_DIR = TEST_DIR / "test_configs"
TINY_TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PRINT_FORMAT = ".1e"


# --- Debugging only --- #
def get_test_model(model_path_or_id: str = TINY_TEST_MODEL):
    return AutoModelForCausalLM.from_pretrained(model_path_or_id)

def get_model_from_config(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    llama_config = LlamaConfig(**config)
    with torch.device("meta"):
        return LlamaForCausalLM(llama_config)

def _sanity_check_mfu(bs, seqlen, model_id=None, config_path=None):
    assert model_id is not None or config_path is not None
    if config_path is not None:
        model = get_model_from_config(config_path)
    else:
        model: PreTrainedModel = get_test_model(model_id)

    model_params = model.num_parameters(exclude_embeddings=True)
    hidden_dim = model.config.hidden_size
    ffn_dim = model.config.intermediate_size
    vocab_size = model.config.vocab_size
    num_layers = model.config.num_hidden_layers

    params_only_flops = palm_transformer_flops_per_iteration_with_attn(
        num_params=model_params,
        num_layers=num_layers,
        d_model=hidden_dim,
        bs=bs,
        seqlen=seqlen,
        include_attn=False,
        activation_checkpointing=False,
        verbose=False,
    )
    params_attn_flops = palm_transformer_flops_per_iteration_with_attn(
        num_params=model_params,
        num_layers=num_layers,
        d_model=hidden_dim,
        bs=bs,
        seqlen=seqlen,
        include_attn=True,
        activation_checkpointing=False,
        verbose=True,
    )
    megatron_flops = megatron_transformer_flops_per_batch(
        bs=bs,
        seqlen=seqlen,
        num_layers=num_layers,
        hidden_size=hidden_dim,
        vocab_size=vocab_size,
        activation_checkpointing=False,
        verbose=True,
    )
    # print(f"params_only_flops: {params_only_flops}")
    # print(f"params_attn_flops: {params_attn_flops}")
    # print(f"megatron_flops: {megatron_flops}")
    rounded_params_flops = round_to_nearest_power_of_10(params_only_flops)
    rounded_params_attn_flops = round_to_nearest_power_of_10(params_attn_flops)
    rounded_megatron_flops = round_to_nearest_power_of_10(megatron_flops)
    print(f"rounded_params_flops: {rounded_params_flops:.1e}")
    print(f"rounded_params_attn_flops: {rounded_params_attn_flops:.1e}")
    print(f"rounded_megatron_flops: {rounded_megatron_flops:.1e}")


@pytest.mark.parametrize("config_path", [TEST_CONFIGS_DIR / "flash_only_config.yml"])
def test_transformer_only_attn_flops(config_path, num_gpus=1):
    global_config = GlobalConfig.from_ymls([config_path])
    bs, seqlen, d_model = get_input_shapes(global_config, num_gpus)
    num_transformer_attn_layers, *_ = parse_layer_types(global_config)

    vocab_size = getattr(global_config, "vocab_size", VOCAB_SIZE)
    activation_checkpointing = False

    # Check attn flops
    attn_flops_ref = megatron_transformer_flops_per_batch(
        bs,
        seqlen,
        num_layers=num_transformer_attn_layers,
        hidden_size=d_model,
        vocab_size=vocab_size,
        activation_checkpointing=activation_checkpointing,
    )
    hyena_flop_counts: HyenaFlopsPerIter = get_hyena_flops(
        global_config,
        num_gpus=num_gpus,
        activation_checkpointing=activation_checkpointing,
    )
    assert hyena_flop_counts.hyena_conv_flops == 0 and hyena_flop_counts.hyena_conv_proj_flops == 0 
    assert hyena_flop_counts.total_flops == attn_flops_ref
    
    print(f"hyena_flops: {hyena_flop_counts.total_flops} -> {round_to_nearest_power_of_10(hyena_flop_counts.total_flops):.1e}")
    print(f"attn_flops_ref: {attn_flops_ref} -> {round_to_nearest_power_of_10(attn_flops_ref):.1e}")
    
    
# TODO: Figure out reference flops counts for cascade
@pytest.mark.parametrize("config_path", [TEST_CONFIGS_DIR / "cascade.yml"])
def test_cascade(config_path, num_gpus=1):
    global_config = GlobalConfig.from_ymls([config_path])
    flop_counts = get_hyena_flops(global_config, num_gpus)
    print(str(flop_counts))


@pytest.mark.parametrize("device_name, dtype, expected_flops", [("h100 hbm3", torch.bfloat16, 989.4e12), ("a100", torch.bfloat16, 312e12)])
def test_available_flops(device_name, dtype, expected_flops):
    
    with patch('torch.cuda.get_device_name') as mock_get_device_name:
        mock_get_device_name.return_value = device_name
        available_flops = get_available_flops(torch.device("cuda"), dtype)
        assert available_flops == expected_flops
        