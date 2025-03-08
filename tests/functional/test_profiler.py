import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

from savanna.arguments import GlobalConfig
from savanna.arguments.global_config import GlobalConfigProfiler
from savanna.profiler import CudaProfiler, EmptyProfiler, NsysProfiler, TorchProfiler

THIS_DIR = Path(__file__).parent
SAVANNA_DIR = THIS_DIR.parent.parent.absolute()
TEST_DIR = SAVANNA_DIR / "tests"
TEST_CONFIGS_DIR = TEST_DIR / "test_configs"

sys.path.insert(0, str(SAVANNA_DIR))
print(f"{THIS_DIR} {SAVANNA_DIR} {TEST_DIR} {TEST_CONFIGS_DIR}")


from tests.functional.train_loop import (
    BACKWARD_LABEL,
    FORWARD_LABEL,
    LOSS_LABEL,
    OPTIMIZER_STEP_LABEL,
    run_train_loop,
    setup_train_loop,
)

TRAIN_LOOP_SCRIPT = Path(__file__).parent / "train_loop.py"
LABEL_SET = {FORWARD_LABEL, BACKWARD_LABEL, LOSS_LABEL, OPTIMIZER_STEP_LABEL}
"""


NOTE: Tests should be run from root (savanna) directory, otherwise paths might 
not be configured correctly.

Usage:
From savanna directory:
python tests/functional/test_profiler.py --model_config tests/test_configs/profiler/{empty,torch,cuda}_profiler.yml
"""

class SmallModel(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def args_to_list(args):
    args_list = [
        "--num_layers",
        str(args.num_layers),
        "--hidden_dim",
        str(args.hidden_dim),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--model_config",
        args.model_config.absolute().as_posix() if args.model_config is not None else None,
    ]
    return args_list


def setup_profiler(global_config: GlobalConfig):
    global_config = GlobalConfig.from_ymls([args.model_config])
    if global_config.should_profile:
        if global_config.profiler_type == "torch":
            profiler = TorchProfiler(
                cpu=global_config.profile_cpu,
                cuda=global_config.profile_cuda,
                profile_memory=global_config.profile_memory,
                with_stack=global_config.profile_with_stack,
                record_shapes=global_config.profile_record_shapes,
                with_flops=global_config.profile_with_flops,
                wait=global_config.profiler_schedule_wait,
                warmup=global_config.profiler_schedule_warmup,
                active=global_config.profiler_schedule_active,
                repeat=global_config.profiler_schedule_repeat,
                output_dir=global_config.profiler_output_dir,
                clean_output_dir=global_config.profiler_clean_output_dir,
                num_rows=global_config.profiler_num_rows,
            )
        elif global_config.profiler_type == "nsys":
            profiler = CudaProfiler(
                emit_nvtx=global_config.emit_nvtx,
                disable_autograd_multithreading=global_config.disable_autograd_multithreading,
                stop_on_exit=global_config.nsys_stop_on_exit,
            )
        else:
            raise NotImplementedError(f"Unknown profiler type: {global_config.profiler_type}")
    else:
        profiler = EmptyProfiler()

    return profiler

def check_labels(names_to_check, labels=LABEL_SET):
    assert all(any(label in name for name in names_to_check) for label in labels)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_config", type=Path, default=TEST_CONFIGS_DIR / "profiler" / "empty_profiler.yml")
    args = parser.parse_args()
    print(args.model_config)
    global_config = GlobalConfig.from_ymls([args.model_config])
    for p in dir(global_config):
        if "profile" in p:
            print(f"{p}: {getattr(global_config, p)}")
            
    if global_config.should_profile and global_config.profiler_type == "nsys":
        with tempfile.TemporaryDirectory() as trace_dir:
            nsys = NsysProfiler(trace_dir=trace_dir)
            report_path, stats_path = nsys.run(
                script=[sys.executable, str(TRAIN_LOOP_SCRIPT)],
                script_args=args_to_list(args),
                stats_report="nvtx_sum",
            )

            print(f"Report path: {report_path}")
            print(f"Stats path: {stats_path}")
            assert report_path is not None
            assert stats_path is not None
            df = pd.read_csv(stats_path)            
            check_labels(df["Range"].unique(), LABEL_SET)
            nvtx_df = df[df["Range"].isin(LABEL_SET)]
            assert all(nvtx_df["Instances"].map(int) == global_config.nsys_num_steps)

    else:
        model, inputs, optimizer = setup_train_loop(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=1e-3,
            device="cuda",
        )
        profiler = setup_profiler(global_config)
        run_train_loop(model, inputs, optimizer, profiler)
        if global_config.should_profile and global_config.profiler_type == "torch":
            assert profiler.profiler is not None
            assert hasattr(profiler, "key_averages")
            event_avgs = profiler.key_averages()
            event_names = set([e.key for e in event_avgs])
            label_events = [e for e in event_avgs if any(label in e.key for label in LABEL_SET)]
            global_config: GlobalConfigProfiler
            check_labels(event_names, LABEL_SET)
            assert all(e.count == global_config.profiler_schedule_active * global_config.profiler_schedule_repeat for e in label_events)
    test_name = str(args.model_config.stem).upper()

    print(f"{test_name} test passed!")