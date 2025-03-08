"""
Copyright (c) Michael Poli
"""

import torch 
import torch.nn as nn

# try importing fa2 and fa3


class EmbeddingOperator(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

class MHAOperator(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

class HyenaOperator(nn.Module):
    def __init__(
            self, 
            dim: int, 
            filter_L: int,
            filter_type: str,
            L_cache: int,
            order: int,
        ):
        super().__init__()

    def forward(self, x):
 
        np = query_layer.shape[-2]
        L = query_layer.shape[0]
        query = rearrange(query_layer, "sq b np hn -> b (np hn) sq")
        key = rearrange(key_layer, "sq b np hn -> b (np hn) sq")
        value = rearrange(value_layer, "sq b np hn -> b (np hn) sq")

        q, k, v = query[..., :L], key[..., :L], value[..., :L]

        if self.use_medium_hyena:
            h = self.filter(self.hyena_medium_conv_len)
        else:
            h = self.filter(L)
            
        if type(h) == tuple:
            h = h[0]

        if self.use_slow_heads:
            return self.multihead_forward(q, k, v, h)

        elif self.use_long_conv1d:
            h = h.repeat_interleave(self.head_dim, dim=0)
            z = k * v

            z = (
                F.conv1d(z, h[:, None].flip(-1), padding=L - 1, groups=v.shape[1])[
                    ..., :L
                ]
                + self.long_conv_bias.unsqueeze(-1) * z
            )
            z = z.to(v.dtype)
            z = q * z
        else:
            h = h.repeat_interleave(self.head_dim, dim=0)

            # breakpoint()
            if self.global_config.use_flashfft:
                # squeeze h dim (kernel), to get rid of leading 1 dim
                h = h.squeeze(0)                
                z = self.fftconv_fn(v, h, k, q)
            else: 
                z = k * v
                with torch.autocast("cuda"):
                    z = fftconv_func(
                        z.to(torch.float32),
                        h.to(torch.float32),
                        self.long_conv_bias,
                        None,
                        gelu=False,
                        bidirectional=self.bidirectional,
                    )
                    z = z.to(v.dtype)
                z = q * z

        return rearrange(z, "b (np hn) sq -> b np sq hn", np=np)     

class GLUOperator(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

class StripedHyenaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass
        # This is a pre-norm, operator, residual block containing two operators i.e.,
        # z = x + norm(op(x))
    
    def forward(self, x):
        pass


class StripedHyenaBackbone(PipelineModule, nn.Module):
    def __init__(
        self,
        config,
    ):
        self.config = config
