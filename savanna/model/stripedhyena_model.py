import torch
import torch.nn as nn
from savanna.model.activations import get_activation
from savanna.model.operators.word_embeddings import EmbeddingPipe, SoftEmbedding
from savanna.model.operators.hyena.hyena import ParallelHyenaOperator
from savanna.mpu.initialize import get_fp8_sync_group
from savanna.linear import FlexLinear

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    from savanna.model.tengine import (
        TENorm,
        TELayerNormColumnParallelLinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        set_format_recipe,
    )
except:
    te = None
    print("WARNING: transformer_engine not installed. Using default recipe.")


# block performs the following operations:
# 1. LayerNorm + Mixer, which can be either Hyena or MHA
# 2. Residual connection
# 3. Gated MLP, with first input layer using Fused norm and linear
# 4. Residual connection


def get_block(global_config):
    if global_config.block_type == "mha":
        return Block(global_config)
    elif global_config.block_type == "hyena":
        return Block(global_config)
    else:
        raise Exception("Only mha and hyena are supported block types")


class Block(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, mixer_cls: str, mlp_cls: str, init_method):
        super().__init__()

        block_number = self.global_config.layer_number  # TODO: standardize this
        init_method = self.global_config.init_method

        self.pre_norm_mixer = TENorm()
        self.pre_norm_mlp = TENorm()
        self.mixer = ParallelHyenaOperator(init_method, block_number)
        self.mlp = GatedMLP()

    def forward(self, x):
        z = self.mha(self.pre_norm_mixer(x)) + x
        y = self.mlp(self.pre_norm_mlp(z)) + z
        return y


class StripedHyena(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.global_config = global_config

        self.num_layers = self.global_config.num_layers
        self.hidden_size = self.global_config.hidden_size
        self.input_emb = EmbeddingPipe(global_config)
        self.output_emb = EmbeddingPipe(global_config)
        self.blocks = nn.ModuleList

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(get_block(global_config))

        def forward(self, x):
            x = self.input_emb(x)
            for block in self.blocks:
                x = block(x)
            x = self.output_emb(x)
            return x


if __name__ == "__main__":
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    from savanna.model.block import ParallelGLU
    from savanna.initialize import initialize_megatron
    from savanna.model.init_functions import xavier_normal_init_method
    from copy import deepcopy

    # Set dimensions.
    in_features = 768
    out_features = 3072
    hidden_size = 2048

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    config = {
        "precision": "fp8",
        "activation": "gelu",
        "hidden_size": hidden_size,
        "lazy_mpu_init": False,
        "local_rank": 0,
        "rank": 0,
        "world_size": 8,
        "pipe_parallel_size": 1,
        "model_parallel_size": 1,
        "seed": 1234,
        "num_layers": 12,
        "checkpoint_num_layers": 1,
        "partition_activations": False,
        "contiguous_checkpointing": False,
        "checkpoint_in_cpu": False,
        "synchronize_each_layer": False,
        "profile_backward": False,
        "params_dtype": "bf16",
    }
    # turn dictionary into something that can be called with dot notation
    config = dotdict(config)

    # initialize model parallel group
    initialize_megatron(config)

    init_method = xavier_normal_init_method(False, 1.0)
    output_layer_init_method = xavier_normal_init_method(False, 1.0)
    l = ParallelGLU(config, init_method, output_layer_init_method)

    x = torch.randn(8, hidden_size, device="cuda")
    x = x.to(torch.bfloat16)
    l = l.cuda()
    y, _ = l(x)
    print(y.shape)

    fp8_format, fp8_recipe = set_format_recipe(config)
    config.use_fp8_linears = True
    l_flex = FlexLinear(
        input_size=hidden_size,
        output_size=hidden_size,
        config=config,
        init_method=init_method,
        parallel_mode="column",
    )
    weight = l_flex.weight
    l_flex = l_flex.cuda()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=get_fp8_sync_group()):
        y_flex, _ = l_flex(x)

    # MP: not copying the input triggers same error as two passes through the model
    # with FP8 enabled?
    x = deepcopy(x)

    config.use_fp8_linears = False
    l_flex_no_fp8 = FlexLinear(
        input_size=hidden_size,
        output_size=hidden_size,
        config=config,
        init_method=init_method,
        parallel_mode="column",
    )
    l_flex_no_fp8.weight = weight
    y_flex_no_fp8, _ = l_flex_no_fp8(x)

    import pdb

    pdb.set_trace()
    assert torch.allclose(y_flex, y_flex_no_fp8)

    # # Initialize model and inputs.
    # model = te.Linear(in_features, out_features, bias=True)
    # inp = torch.randn(hidden_size, in_features, device="cuda")

    # # Create an FP8 recipe. Note: All input args are optional.
    # fp8_recipe = recipe.DelayedScaling(
    #     margin=0, interval=1, fp8_format=recipe.Format.E4M3
    # )

    # # Enable autocasting for the forward pass
    # with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    #     out = model(inp)

    # loss = out.sum()
    # loss.backward()
    # print("Done")
