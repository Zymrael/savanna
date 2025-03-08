import torch
from torch.nn import LayerNorm as LayerNorm
import transformer_engine.pytorch as te
from savanna.utils import ALLOC_DEVICE


def _get_torch_norm(global_config):
    if global_config.norm == "rmsnorm":
        norm = RMSNorm
        eps = global_config.rms_norm_epsilon
    elif global_config.norm == "layernorm":
        eps = global_config.layernorm_epsilon
        norm = LayerNorm
    elif global_config.norm == "scalenorm":
        eps = global_config.scalenorm_epsilon
        norm = ScaleNorm
    else:
        raise ValueError(f"norm {global_config.norm} not recognized")
    return norm, eps


def _get_te_norm(global_config):
    if global_config.norm == "rmsnorm":
        norm = te.RMSNorm
        eps = global_config.rms_norm_epsilon
    elif global_config.norm == "layernorm":
        eps = global_config.layernorm_epsilon
        norm = te.LayerNorm
    elif global_config.norm == "scalenorm":
        eps = global_config.scalenorm_epsilon
        norm = ScaleNorm
    else:
        raise ValueError(f"norm {global_config.norm} not recognized")
    return norm, eps


def get_norm(global_config):
    if global_config.use_fp8_norm:
        return _get_te_norm(global_config)
    else:
        return _get_torch_norm(global_config)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim, device=ALLOC_DEVICE))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim, device=ALLOC_DEVICE))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        # We support mixed types for scale/offset, though currently
        # deepspeed casts all params to bf16 anyway.
        if self.bias:
            out = self.scale * x_normed + self.offset
        else:
            out = self.scale * x_normed
        return out.type_as(x)


class ScaleNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g
