import torch.nn as nn

from savanna import mpu, print_rank_0
from savanna.linear import FlexLinear
from savanna.model.activations import get_activation
from savanna.mpu.initialize import get_fp8_sync_group
from savanna.utils import FP8_SHAPE, pad_to_multiple

try:
    import transformer_engine.pytorch as te

    from savanna.model.tengine import set_format_recipe
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
        seq_dim=1, # Needed after refactor
    ):
        super().__init__()

        #@jeromeku add these for sequence parallel config and checks
        self.global_config = global_config
        self.pad_mlp_weights = global_config.pad_mlp_weights
        self.seq_dim = seq_dim
        self.use_fp8 = global_config.use_fp8_mlp_projections or global_config.use_fp8_linears
        self.disable_fp8_w3 = global_config.disable_fp8_w3
        self.model_parallel_size = global_config.model_parallel_size
        self.should_permute = self.use_fp8 and self.global_config.sequence_parallel and self.global_config.permute_glu

        if self.use_fp8:
            self.fp8_format, self.fp8_recipe = set_format_recipe(global_config)

        self.activation_func = get_activation(global_config)
        self.activation_type = global_config.activation

        self.multiple_of = multiple_of

        if global_config.ff_dim is not None:
            ff_dim = global_config.ff_dim
            # print_rank_0(f"DEBUG::ParallelGLU::ff_dim: manually set ff_dim to {ff_dim}")
        else:
            ff_dim = int(2 * global_config.hidden_size * 4 / 3)
            ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
            # print_rank_0(f"DEBUG::ParallelGLU::ff_dim:before padding {ff_dim}")
            if self.pad_mlp_weights:
                # NOTE: @jeromeku
                # Pad ff_dim to be divisible by 16
                # Corresponds to input_dim of w3, output_dim of w1 and w2
                # Needed when model_parallel_size > 8, needed for extending 40B at context lengths 256K+
                # Each mp partition needs to have a multiple of 16 dim
                ff_dim = pad_to_multiple(ff_dim, FP8_SHAPE[1] * self.model_parallel_size)
            # print_rank_0(f"DEBUG::ParallelGLU::ff_dim:after padding {ff_dim}")

        
        #@jeromeku need to 
        extra_kwargs = {} if self.use_fp8 else {"seq_dim": seq_dim}
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
            **extra_kwargs,
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
            **extra_kwargs,
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
            use_fp8=self.use_fp8 and not self.disable_fp8_w3,
            **extra_kwargs,
        )
        print_rank_0(f"DEBUG::ParallelGLU::w1: {type(self.w1)}")
        print_rank_0(f"DEBUG::ParallelGLU::w2: {type(self.w2)}")
        print_rank_0(f"DEBUG::ParallelGLU::w3: {type(self.w3)}")

    def forward_func(self, hidden_states):
        #@jerome ku hidden_states [B, L, H]
        # When using SP, need to have L as the first dimension
        if self.should_permute:
            #print_rank_0("Permuting hidden states for GLU from {} to {}".format(hidden_states.shape, hidden_states.permute(1, 0, 2).shape))
            hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        
        if self.global_config.debug_print:  
            print_rank_0(f"DEBUG::ParallelGLU::hidden_states: {hidden_states.shape}")
        
        w1_out, _ = self.w1(hidden_states)
        
        if self.global_config.debug_print:
            print_rank_0(f"DEBUG::ParallelGLU::w1_out: {w1_out.shape}")
        
        w2_out, _ = self.w2(hidden_states)
        
        out = self.w3(self.activation_func(w1_out) * w2_out)
        
        dense_out, dense_bias = out
        
        if self.global_config.debug_print:
            print_rank_0(f"DEBUG::ParallelGLU::dense_out: {dense_out.shape}")
        
        if self.should_permute:
            #print_rank_0("Permuting GLU output from {} to {}".format(dense_out.shape, dense_out.permute(1, 0, 2).shape))
            #[L, B, H] -> [B, L, H]
            dense_out = dense_out.permute(1, 0, 2).contiguous()
        
        #@jeromeku check that output is scattered along correct dim (seq_dim) when using SP
        seq_len = self.global_config.seq_length // mpu.get_sequence_parallel_world_size()
        #print_rank_0(f"DEBUG::ParallelGLU::seq_len: {seq_len}")
       
        if not self.global_config.enable_haystack:
            if self.global_config.sequence_parallel:
                pass
                # B, L, H = dense_out.shape
                # assert L == dense_out.shape[self.seq_dim]
                # assert L == (seq_len // self.global_config.model_parallel_size), f"{L=} != {(seq_len // self.global_config.model_parallel_size)=}"
            else:
                pass
                # assert dense_out.shape[self.seq_dim] == seq_len, f"{dense_out.shape[self.seq_dim]=} != {seq_len=}"
        
        return (dense_out, dense_bias)
    
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
