import torch

def get_dtype_from_string(dtype_str):
    if type(dtype_str) == str:
        if dtype_str == "float32" or dtype_str == "fp32":
            return torch.float32
        elif dtype_str == "float16" or dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bfloat16" or dtype_str == "bf16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unrecognized dtype {dtype_str}")
    else:
        return dtype_str