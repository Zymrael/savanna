# Copyright (c) 2024, Michael Poli.

# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli

from typing import Callable

import torch

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    from transformer_engine.pytorch import Linear
except:
    print("WARNING: transformer_engine not installed. Using default recipe.")
    raise NotImplementedError

def set_format_recipe():
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_format, fp8_recipe


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
        init_method: Callable,
        bias: bool = False,
        skip_bias_add: bool = False,
        use_fp8: bool = False,
        params_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.use_fp8 = use_fp8
        
        if use_fp8:
            self.fp8_format, self.fp8_recipe = set_format_recipe()

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            init_method=init_method,
            params_dtype=params_dtype,
            bias=bias,
            return_bias=self.te_return_bias,
            **kwargs,
        )

    def forward(self, x):

        if self.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                out = super().forward(x)
        else:
            out = super().forward(x)

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None
    
