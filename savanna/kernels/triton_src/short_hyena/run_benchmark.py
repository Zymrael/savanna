import argparse
import itertools
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from einops import rearrange
import yaml
from triton.testing import do_bench

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
    get_autotuned_kernel_configs,
)

# Enable import of benchmark utils
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent))


from savanna.kernels.triton_src.short_hyena.benchmark.utils import (
    BenchResult,
    LoggingContext,
    # ShapeConfig,
    Tee,
    generate_configs,
    post_process_autotune_results,
    postprocess,
    print_delimiter,
    setup_inputs,
)
from savanna.kernels.triton_src.short_hyena.interface import hyena_mlp
from savanna.kernels.triton_src.short_hyena.src.kernel_utils import (
    ShortHyenaOperatorKernelConfig,
    PostConvKernelConfig,
    PreConvKernelConfig,
    ShapeConfig,
)


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


def run_bench_single(
    shape_config,
    dtype,
    fwd_config: ShortHyenaOperatorKernelConfig = None,
    bwd_config: ShortHyenaOperatorKernelConfig = None,
    autotune=False,
    repeats=100,
    verbose=True,
) -> BenchResult:
    test_inputs, ref_inputs, ref_causal_inputs = setup_inputs(shape_config, dtype)

    q_ref, k_ref, v_ref, w_ref = ref_inputs
    q_ref_causal, k_ref_causal, v_ref_causal, w_ref_causal = ref_causal_inputs
    q, k, v, w = test_inputs

    def _ref_fn_fwd(q, k, v, w, use_causal_conv, contiguous=True):
        return ref_hyena_mlp(
            q,
            k,
            v,
            w,
            use_causal_conv=use_causal_conv,
            contiguous=contiguous,
            repeat_interleave=True,
        )

    _y = (
        _ref_fn_fwd(q_ref, k_ref, v_ref, w_ref, use_causal_conv=False, contiguous=True)
        .detach()
        .clone()
    )
    dy = torch.randn_like(_y)

    def ref_fwd_bwd(q, k, v, w, dy, use_causal_conv, contiguous=True):
        y = _ref_fn_fwd(q, k, v, w, use_causal_conv, contiguous)
        y.backward(dy)

    ref_bench_fn = lambda: ref_fwd_bwd(
        q_ref, k_ref, v_ref, w_ref, dy, use_causal_conv=False, contiguous=True
    )
    ref_causal_bench_fn = lambda: ref_fwd_bwd(
        q_ref_causal,
        k_ref_causal,
        v_ref_causal,
        w_ref_causal,
        dy,
        use_causal_conv=True,
        contiguous=True,
    )

    def triton_fwd_bwd(q, k, v, w, dy, fwd_config, bwd_config, use_causal_conv):
        y = hyena_mlp(
            q=q,
            k=k,
            v=v,
            w=w,
            autotune=autotune,
            fwd_kernel_cfg=fwd_config,
            bwd_kernel_cfg=bwd_config,
            use_causal_conv=use_causal_conv,
            repeat_interleave=True,
        )
        y.backward(dy)

    triton_bench_fn = lambda: triton_fwd_bwd(
        q, k, v, w, dy, fwd_config, bwd_config, use_causal_conv=False
    )
    # Fwd benchmark
    triton_bench_fn_causal = lambda: triton_fwd_bwd(
        q, k, v, w, dy, fwd_config, bwd_config, use_causal_conv=True
    )

    triton_t = do_bench(triton_bench_fn, rep=repeats, grad_to_none=[q, k, v, w])
    ref_t = do_bench(
        ref_bench_fn, rep=repeats, grad_to_none=[q_ref, k_ref, v_ref, w_ref]
    )
    speedup = ref_t / triton_t

    if verbose:
        print(f"{shape_config=}, {fwd_config=}, {bwd_config=}", flush=True)
        print(f"  {triton_t=}, {ref_t=} -> {speedup=}", flush=True)

    if shape_config.kernel_size <= 4:
        ref_t_causal = do_bench(
            ref_causal_bench_fn,
            rep=repeats,
            grad_to_none=[q_ref_causal, k_ref_causal, v_ref_causal, w_ref_causal],
        )
        triton_t_causal = do_bench(
            triton_bench_fn_causal, rep=repeats, grad_to_none=[q, k, v, w]
        )
        speedup_causal = ref_t_causal / triton_t_causal
        if verbose:
            print(
                f"  {triton_t_causal=}, {ref_t_causal=} -> {speedup_causal=}",
                flush=True,
            )
    else:
        triton_t_causal = None
        ref_t_causal = None
        speedup_causal = None

    res = BenchResult(
        triton_time=triton_t,
        ref_time=ref_t,
        speedup=speedup,
        triton_causal_time=triton_t_causal,
        ref_causal_time=ref_t_causal,
        speedup_causal=speedup_causal,
    )

    return res


def run_bench(
    shape_configs: List[ShapeConfig],
    dtype: torch.dtype,
    kernel_configs: List[Tuple[PreConvKernelConfig, PostConvKernelConfig]],
    autotune: bool = True,
    repeats=100,
    verbose=True,
) -> Dict[
    ShapeConfig, List[Tuple[ShortHyenaOperatorKernelConfig, ShortHyenaOperatorKernelConfig, BenchResult]]
]:
    results = defaultdict(list)
    failed_configs = defaultdict(list)

    for shape_config in shape_configs:
        print(f"Running {shape_config=}", flush=True)

        for fwd_pre_conv_config, fwd_post_conv_config in kernel_configs:
            for bwd_pre_conv_config, bwd_post_conv_config in kernel_configs:
                fwd_config = ShortHyenaOperatorKernelConfig(
                    pre_conv_kernel_config=fwd_pre_conv_config,
                    post_conv_kernel_config=fwd_post_conv_config,
                )
                bwd_config = ShortHyenaOperatorKernelConfig(
                    pre_conv_kernel_config=bwd_pre_conv_config,
                    post_conv_kernel_config=bwd_post_conv_config,
                )

                try:
                    res = run_bench_single(
                        shape_config,
                        dtype,
                        fwd_config,
                        bwd_config,
                        repeats=repeats,
                        verbose=verbose,
                    )

                except Exception as e:
                    print(
                        f"Skipping {shape_config=}, {fwd_config=}, {bwd_config=} due to {e}",
                        flush=True,
                    )
                    failed_configs[shape_config].append((fwd_config, bwd_config))
                else:
                    results[shape_config].append((fwd_config, bwd_config, res))
    return results, failed_configs


def setup_bench(config):
    shape_configs, pre_conv_configs, post_conv_configs = generate_configs(config)
    dtype_str = config["dtype"]
    dtype = getattr(torch, dtype_str)
    kernel_configs = list(itertools.product(pre_conv_configs, post_conv_configs))

    return shape_configs, dtype, kernel_configs


def setup_config(args):
    config = yaml.safe_load(args.config.read_text())
    for key, value in vars(args).items():
        if (
            value is not None and key not in config
        ):  # Only override if the argument was provided
            config[key] = value

    config["config"] = str(args.config)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    config["save_path"] = os.path.join(config["save_path"], ts)
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    # config["save_path"] = str(config["save_path"].resolve())
    config_path = os.path.join(config["save_path"], "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    return config


def setup_logging(config):
    log_file = open(os.path.join(config["save_path"], "output.log"), "w")

    # Redirect stdout and stderr to both the terminal and the log file
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    log_file.close()


def run_autotune_bench(
    shape_configs, dtype, repeats=100, verbose=True
) -> List[Tuple[ShapeConfig, Dict, BenchResult]]:
    results = []
    for shape_config in shape_configs:
        print(f"Running {shape_config=}", flush=True)
        res = run_bench_single(shape_config, dtype, autotune=True, repeats=repeats)
        if verbose:
            print(res, flush=True)

        # Extract best config
        best_configs = get_autotuned_kernel_configs()
        results.append((shape_config, best_configs, res))
    return results


def main(args):
    # Setup
    config = setup_config(args)
    shape_configs, dtype, kernel_configs = setup_bench(config)

    pd.set_option("display.expand_frame_repr", False)
    # print the entire data frame
    pd_print_full_ctx = pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    )
    with LoggingContext(os.path.join(config["save_path"], "output.log")):
        # Run
        start = time.time()
        print_delimiter()

        with pd_print_full_ctx:
            if args.autotune:
                print(
                    f"Running autotuning for {len(shape_configs)} shape configs",
                    flush=True,
                )
                results = run_autotune_bench(
                    shape_configs, dtype, repeats=config["repeats"], verbose=True
                )
                df = post_process_autotune_results(results, config["save_path"])
                print_delimiter()
                print("Autotune results:")
                print(df)
            else:
                print(
                    f"Running {len(shape_configs)} shape configs across {len(kernel_configs)} kernel configs",
                    flush=True,
                )
                results, failed_configs = run_bench(
                    shape_configs,
                    dtype,
                    kernel_configs,
                    repeats=config["repeats"],
                    verbose=True,
                )

                end = time.time()
                elapsed = end - start
                print(f"Benchmarking took {elapsed:.0f} seconds", flush=True)
                # Postprocess

                results_df, failed_df = postprocess(
                    results, failed_configs, config["save_path"]
                )
                print_delimiter()
                print("Results:")
                print(results_df)

                if failed_df is not None:
                    print("Failed configs:")
                    print(failed_df)

        print(f"Saved results to {config['save_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=BENCH_DIR / "configs/debug.yaml")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--save_path", type=Path, default=BENCH_DIR / "bench_results")
    args = parser.parse_args()
    main(args)
