import torch
import torch.nn as nn
from savanna.linear import FlexLinear
from savanna.model.activations import get_activation
from savanna.mpu.initialize import get_fp8_sync_group
from savanna import mpu

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    from savanna.model.tengine import (
        TELayerNormColumnParallelLinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        set_format_recipe,
    )
except:
    te = None
    print("WARNING: transformer_engine not installed. Using default recipe.")


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Does not support fp8 yet.
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

        if (self.activation_type == "gelu" and self.bias_gelu_fusion) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelGLU(nn.Module):
    def __init__(
        self,
        global_config,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=64,
    ):
        super().__init__()
        self.use_fp8 = global_config.use_fp8_mlp_projections or global_config.use_fp8_linears
        if self.use_fp8:
            self.fp8_format, self.fp8_recipe = set_format_recipe(global_config)

        self.activation_func = get_activation(global_config)
        self.activation_type = global_config.activation

        self.multiple_of = multiple_of

        ff_dim = int(2 * global_config.hidden_size * 4 / 3)
        ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        # print(ff_dim, global_config.hidden_size)

        # print(init_method, output_layer_init_method)

        self.w1 = FlexLinear(
            input_size=global_config.hidden_size,
            output_size=ff_dim,
            config=global_config,
            parallel_mode="column",
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            gather_output=False,
            use_fp8=self.use_fp8,
        )
        self.w2 = FlexLinear(
            input_size=global_config.hidden_size,
            output_size=ff_dim,
            config=global_config,
            parallel_mode="column",
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            gather_output=False,
            use_fp8=self.use_fp8,
        )
        self.w3 = FlexLinear(
            input_size=ff_dim,
            output_size=global_config.hidden_size,
            config=global_config,
            parallel_mode="row",
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=False,
            input_is_parallel=mpu.initialize.get_model_parallel_world_size() > 1,
            use_fp8=self.use_fp8,
        )

    def forward_func(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w2_out, _ = self.w2(hidden_states)
        return self.w3(self.activation_func(w1_out) * w2_out)

    def forward(self, hidden_states):

        if self.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe, fp8_group=get_fp8_sync_group()):
                return self.forward_func(hidden_states)
        else:
            return self.forward_func(hidden_states)


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
        seq_dim=1
    ):
        super().__init__()
        parallelism = global_config.output_layer_parallelism
        # This is beccause ColumnParallelLinear has functionality to all gather from the sequence parallel region
        # and RowParallelLinear does not
        if global_config.sequence_parallel and not parallelism == "column":
            raise ValueError("output_layer_parallelism=row not supported with sequence parallel")
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
                seq_dim=1
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
