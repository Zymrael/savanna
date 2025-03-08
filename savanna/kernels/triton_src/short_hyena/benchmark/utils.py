import itertools
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch

from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
    ShortHyenaOperatorKernelConfig,
    KernelConfig,
    PostConvKernelConfig,
    PreConvKernelConfig,
    ShapeConfig,
)


@dataclass(frozen=True)
class BenchResult:
    triton_time: float
    ref_time: float
    speedup: float
    triton_causal_time: float = None,
    ref_causal_time: float = None
    speedup_causal: float = None


def _parse_config_into_dataclass(config, datacls, check_all_fields_present=True):
    fields = [f for f in datacls.__dataclass_fields__]
    vals = [config.get(f) for f in fields]
    print(fields, vals)
    if check_all_fields_present:
        assert all(v is not None for v in vals)
    parsed_config = [
        datacls(**dict(zip(fields, vals))) for vals in list(itertools.product(*vals))
    ]
    return parsed_config


def parse_kernel_configs(kernel_configs, config_type=PreConvKernelConfig):
    return _parse_config_into_dataclass(kernel_configs, config_type)


def parse_shapes(shapes):
    return _parse_config_into_dataclass(shapes, ShapeConfig)
    # shape_fields = [f for f in ShapeConfig.__dataclass_fields__]
    # shape_vals = [shapes.get(f) for f in shape_fields]
    # assert all(v is not None for v in shape_vals)
    # parsed_shapes = [ShapeConfig(**dict(zip(shape_fields, vals))) for vals in list(itertools.product(*shape_vals))]
    # return parsed_shapes


def generate_configs(config):
    shape_configs = parse_shapes(config["shapes"])
    pre_conv_kernel_configs = parse_kernel_configs(
        config["pre_conv_kernel_configs"], PreConvKernelConfig
    )
    post_conv_kernel_configs = parse_kernel_configs(
        config["post_conv_kernel_configs"], PostConvKernelConfig
    )
    return shape_configs, pre_conv_kernel_configs, post_conv_kernel_configs

def _unpack_config(config: KernelConfig, prefix=""):
    return {f"{prefix}_{k}": v for k, v in config.to_dict().items()}

def _process_results(results):
    rows = []
    for shape_config in results.keys():
        for fwd_config, bwd_config, res in results[shape_config]:
            fwd_pre_config = fwd_config.pre_conv_kernel_config
            fwd_post_config = fwd_config.post_conv_kernel_config
            bwd_pre_config = bwd_config.pre_conv_kernel_config
            bwd_post_config = bwd_config.post_conv_kernel_config
            
            fwd_pre_config_dict = _unpack_config(fwd_pre_config, "FWD")
            fwd_post_config_dict = _unpack_config(fwd_post_config, "FWD")
            bwd_pre_config_dict = _unpack_config(bwd_pre_config, "BWD")
            bwd_post_config_dict = _unpack_config(bwd_post_config, "BWD")
            row = {
                **asdict(shape_config),
                **fwd_pre_config_dict,
                **fwd_post_config_dict,
                **bwd_pre_config_dict,
                **bwd_post_config_dict,
                **asdict(res),
            }
            rows.append(row)
    results_df = pd.DataFrame(rows)
    return results_df


def _process_failed_configs(failed_configs):
    rows = []
    for shape_config in failed_configs.keys():
        for fwd_config, bwd_config in failed_configs[shape_config]:
            fwd_pre_config = fwd_config.pre_conv_kernel_config
            fwd_post_config = fwd_config.post_conv_kernel_config
            bwd_pre_config = bwd_config.pre_conv_kernel_config
            bwd_post_config = bwd_config.post_conv_kernel_config

            fwd_pre_config_dict = _unpack_config(fwd_pre_config, "FWD")
            fwd_post_config_dict = _unpack_config(fwd_post_config, "FWD")
            bwd_pre_config_dict = _unpack_config(bwd_pre_config, "BWD")
            bwd_post_config_dict = _unpack_config(bwd_post_config, "BWD")
            
            row = {
                **asdict(shape_config),
                **fwd_pre_config_dict,
                **fwd_post_config_dict,
                **bwd_pre_config_dict,
                **bwd_post_config_dict,
                }
            rows.append(row)
    failed_df = pd.DataFrame(rows)
    return failed_df


def postprocess(
    results: Dict[
        ShapeConfig, List[Tuple[ShortHyenaOperatorKernelConfig, ShortHyenaOperatorKernelConfig, BenchResult]]
    ],
    failed_configs: List[Tuple[ShortHyenaOperatorKernelConfig, ShortHyenaOperatorKernelConfig]],
    save_path: str,
) -> pd.DataFrame:
    results_df = _process_results(results)
    results_df.to_csv(os.path.join(save_path, "results.csv"), index=False)

    if len(failed_configs) > 0:
        failed_df = _process_failed_configs(failed_configs)
        failed_df.to_csv(os.path.join(save_path, "failed.csv"), index=False)
    else:
        failed_df = None

    return results_df, failed_df


def post_process_autotune_results(results, save_path):

    def _process_results(results):
        rows = []
        for shape_config, best_config, bench_res in results:
            row = {**asdict(shape_config), **best_config, **asdict(bench_res)}
            rows.append(row)
        return pd.DataFrame(rows)

    results_df = _process_results(results)
    results_df.to_csv(os.path.join(save_path, "autotuned_results.csv"), index=False)

    #Extract kernel configs
    # from hyena_mlp.triton.kernel_utils import get_autotune_configs
    # config_dict = get_autotune_configs()
    # with open(os.path.join(save_path, "autotune_configs.json"), "w") as f:
    #     json.dump(config_dict, f)
    return results_df

def print_delimiter(ch="=", n=80):
    print(f"\n{ch * n}\n", flush=True)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure output is flushed immediately

    def flush(self):
        for f in self.files:
            f.flush()


class LoggingContext:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr

    def __enter__(self):
        self.log_file = open(self.log_file_path, "w")
        sys.stdout = Tee(self.stdout_original, self.log_file)
        sys.stderr = Tee(self.stderr_original, self.log_file)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original
        if self.log_file:
            self.log_file.close()


def clone_and_detach(t: torch.Tensor, requires_grad=True):
    return t.clone().detach().requires_grad_(requires_grad)


def setup_inputs(shape_config: ShapeConfig, dtype):
    bs = shape_config.bs
    seqlen = shape_config.seqlen
    g = shape_config.num_groups
    dg  = shape_config.group_dim
    kernel_size = shape_config.kernel_size

    in_shape = (seqlen, bs, g, dg)
    out_shape = (bs, g, seqlen, dg)
    d = g * dg

    k = torch.randn(*in_shape, dtype=dtype, device="cuda", requires_grad=True)
    q = torch.randn_like(k).requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    w = torch.randn(g, 1, kernel_size, device="cuda", dtype=k.dtype, requires_grad=True)

    test_inputs = (q, k, v, w)
    
    ref_inputs = [clone_and_detach(t) for t in test_inputs]
    ref_causal_inputs = [clone_and_detach(t) for t in test_inputs]
    return test_inputs, ref_inputs, ref_causal_inputs
