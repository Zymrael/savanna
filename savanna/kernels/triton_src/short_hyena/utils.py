import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
    
from einops import rearrange



def torch_conv(x, weight):
    _, d_model, L = x.shape
    kernel_size = weight.shape[-1]

    y = F.conv1d(
        x,
        weight,
        bias=None,
        stride=1,
        padding=kernel_size - 1,
        groups=d_model,
    )
    y = y[..., :L]
    
    return y

def causal_conv(x, weight):
    weight = weight.squeeze()
    y = causal_conv1d_fn(x, weight, bias=None, activation=None)
    return y


def ref_hyena_mlp(q, k, v, w, contiguous=False, use_causal_conv=False, debug=False, repeat_interleave=False):
    seqlen, bs, g, dg = q.shape
    d = g * dg
    
    if repeat_interleave:
        assert g == w.shape[0]
        w = w.repeat_interleave(dg, dim=0)
        w.retain_grad()
        assert w.shape[0] == d
    
    # print(f"{w.shape=}")
    
    kv = rearrange(k * v, "seqlen bs np hn -> bs (np hn) seqlen")
    q_permuted = rearrange(q, "seqlen bs np hn -> bs (np hn) seqlen")

    if use_causal_conv and causal_conv1d_fn is not None:
        conv_out = causal_conv(kv, w)
    else:
        conv_out = torch_conv(kv, w)

    out = conv_out * q_permuted
    out = rearrange(
        out, "b (g dg) sq -> b g sq dg", g=g
    )  #    rearrange(z, "b (np hn) sq -> b np sq hn", np=np)

    if contiguous:
        out = out.contiguous()

    if debug:
        return out, w
    return out