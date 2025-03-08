import torch 
import torch.nn.functional as F

from savanna.kernels.triton_src.short_hyena.interface import ShortHyenaOperator, run_short_hyena
from savanna.kernels.triton_src.short_hyena.src.kernel_utils import ShortHyenaOperatorKernelConfig, PreConvKernelConfig, PostConvKernelConfig
from savanna.kernels.triton_src.short_hyena.benchmark.utils import setup_inputs
from savanna.kernels.triton_src.short_hyena.src.kernel_utils import ShapeConfig
from savanna.model.operators.hyena.hyena import ParallelShortHyenaOperator, ParallelCausalDepthwiseConv1d
from savanna.model.init_functions import xavier_normal_init_method
from einops import rearrange

import pytest 

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]

def causal_conv(x, weight):
    weight = weight.squeeze()
    y = causal_conv1d_fn(x, weight, bias=None, activation=None)
    return y


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

    if use_causal_conv:
        out = out.contiguous()

    if debug:
        return out, w
    return out


def prepare_inputs(seqlen, bs, g, dg, kernel_size):
    in_shape = (seqlen, bs, g, dg)
    out_shape = (bs, g, seqlen, dg)

    q = torch.randn(*in_shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    w = torch.randn(g, 1, kernel_size, device="cuda", requires_grad=True)
    return (q, k, v, w)

def prepare_kernel_configs():
    pre_conv_kernel_config = PreConvKernelConfig(
        BLOCK_M=256,
        BLOCK_N=256,
        NUM_PIPELINE_STAGES=1,
        num_warps=4,
        num_ctas=1,
    )
    post_conv_kernel_config = PostConvKernelConfig(
        BLOCK_M=128,
        BLOCK_N=128,
        NUM_PIPELINE_STAGES=1,
        num_warps=4,
        num_ctas=1,
    )
    
    fwd_kernel_cfg = ShortHyenaOperatorKernelConfig(
        pre_conv_kernel_config=pre_conv_kernel_config,
        post_conv_kernel_config=post_conv_kernel_config,
    )
    bwd_kernel_cfg = ShortHyenaOperatorKernelConfig(
        pre_conv_kernel_config=pre_conv_kernel_config,
        post_conv_kernel_config=post_conv_kernel_config,
    )
    return fwd_kernel_cfg, bwd_kernel_cfg


def prepare_ref_config(seqlen, bs, d_model, num_groups, kernel_size):
    config = {
        "hidden_size": d_model,
        "use_cgcg_mlp": False,
        "use_cgcg_short": False,
        "is_mlp": True,
        "hyena_mlp_len": kernel_size,
        "short_conv_L": kernel_size,
        "hyena_short_conv_len": kernel_size,
        "hyena_mlp_pregate": True,
        "hyena_mlp_postgate": True,
        "hyena_short_conv_postgate": True,
        "hyena_short_conv_pregate": True,
        "num_groups_hyena_mlp": num_groups,
        "num_groups_hyena_short": num_groups,
        "hyena_short_conv_len": kernel_size,
        "use_custom_hyena_mlp_kernel": False,
        "use_custom_hyena_short_kernel": False,
        "hyena_width_expansion": 1,
        "conv_proj_bias": True,
        "params_dtype": torch.bfloat16,
    }
    config = DotDict(config)
    return config

    
@pytest.mark.parametrize("seqlen, bs, group_size, d_model, kernel_size", [(8192, 16, 16, 768, 7)])
def test_short_hyena_kernel_fwd_bwd(seqlen, bs, group_size, d_model, kernel_size):
    assert d_model % group_size == 0
    
    shape_config = ShapeConfig(bs=bs, seqlen=seqlen, num_groups=d_model // group_size, d_model=d_model, kernel_size=kernel_size)
    
    test_inputs, ref_inputs, ref_causal_inputs = setup_inputs(shape_config, dtype=torch.bfloat16)
    ref_config = prepare_ref_config(seqlen, bs, d_model, d_model // group_size, kernel_size)
    fwd_kernel_cfg, bwd_kernel_cfg = prepare_kernel_configs()
    
    q_torch, k_torch, v_torch, w_torch = ref_inputs 
    q, k, v, w = test_inputs 
    
    custom_out = ShortHyenaOperator.apply(
        q, 
        k, 
        v, 
        w,
        True,
        False,
        False,
        fwd_kernel_cfg,
        bwd_kernel_cfg,
    )
    
    ref_op = ParallelShortHyenaOperator(
        hidden_size=d_model,
        global_config=ref_config,
        init_method=xavier_normal_init_method,
        short_conv_class=ParallelCausalDepthwiseConv1d,
        use_fast_causal_conv=False,
        is_mlp=False,
        local_init=True,
    )
    ref_op.short_conv.short_conv_weight = torch.nn.Parameter(w)

    torch_out = ref_op(q_torch, k_torch, v_torch)

    assert torch.allclose(custom_out, torch_out, atol=1e-5)

    grad = torch.randn_like(custom_out)
    
    custom_out.backward(grad)
    torch_out.backward(grad)
    
    assert torch.allclose(q.grad, q_torch.grad, atol=1e-5)
    assert torch.allclose(k.grad, k_torch.grad, atol=1e-5)
    assert torch.allclose(v.grad, v_torch.grad, atol=1e-5)


