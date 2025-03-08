import torch
import torch.nn as nn
from pkg_resources import packaging


from savanna.mpu.layers import RowParallelLinear, ColumnParallelLinear
from savanna.utils import get_dtype_from_string

from importlib.metadata import version
from typing import Callable

try:
    from savanna.model.tengine import TELinear
except ImportError:
    TELinear = None


class FlexLinear:
    """
    Megatron and Transformer Engine linear layer compatible with fp8, bf16, fp16 and fp32
    """

    def __new__(
        self,
        input_size,
        output_size,
        config,
        parallel_mode: str,
        bias: bool = False,
        skip_bias_add: bool = True,
        use_fp8: bool = False,
        input_is_parallel=False,  # for row parallel
        gather_output: bool = True,  # for column parallel
        parallel_output: bool = False,  # for row parallel
        **kwargs
    ):
        # use_fp8 = config.use_fp8_linears
        self.config = config
        instance = None

        if use_fp8:
            instance = TELinear(
                input_size=input_size,
                output_size=output_size,
                config=self.config,
                parallel_mode=parallel_mode,
                bias=bias,
                skip_bias_add=skip_bias_add,
                **kwargs,
            )
        else:
            # TODO: unify raw Megatron TP Linears with TE Linears
            if parallel_mode == "column":
                instance = ColumnParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    global_config=self.config,
                    bias=bias,
                    gather_output=gather_output,
                    skip_bias_add=skip_bias_add,
                    **kwargs,
                )
            elif parallel_mode == "row":
                instance = RowParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    global_config=self.config,
                    bias=bias,
                    skip_bias_add=skip_bias_add,
                    input_is_parallel=input_is_parallel,
                    parallel_output=parallel_output,
                    **kwargs,
                )

        return instance
