import argparse
import sys
from pathlib import Path

import torch

FUNCTIONAL_TEST_DIR = Path(__file__).parent
TEST_DIR = FUNCTIONAL_TEST_DIR.parent
TEST_CONFIG_DIR = TEST_DIR / "test_configs"
SAVANNA_DIR = TEST_DIR.parent
LABEL_PREFIX = "##"

FORWARD_LABEL = f"{LABEL_PREFIX}FORWARD"
BACKWARD_LABEL = f"{LABEL_PREFIX}BACKWARD"
LOSS_LABEL = f"{LABEL_PREFIX}LOSS"
OPTIMIZER_STEP_LABEL = f"{LABEL_PREFIX}OPTIMIZER_STEP"

sys.path.insert(0, str(SAVANNA_DIR))


from savanna.arguments import GlobalConfig
from savanna.profiler import CudaProfiler, EmptyProfiler, TorchProfiler


        # return result
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

def run_train_loop(model, inputs, optimizer, profiler):
    profiler.start()
    for input in inputs:
        profiler.step()

        optimizer.zero_grad()
        with profiler.mark(FORWARD_LABEL):
            y = model(input)
        with profiler.mark(LOSS_LABEL):
            loss = y.sum()
        with profiler.mark(BACKWARD_LABEL):
            loss.backward()
        with profiler.mark(OPTIMIZER_STEP_LABEL):
            optimizer.step()

    profiler.stop()

def setup_train_loop(num_layers, hidden_dim, batch_size, num_epochs, lr=1e-3, device="cuda"):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    model = SmallModel(num_layers=num_layers, hidden_dim=hidden_dim).to(device)
    inputs = [torch.randn(batch_size, hidden_dim, device=device) for _ in range(num_epochs)]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, inputs, optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_config", type=Path, default=None)
    args = parser.parse_args()

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
                debug=True,
                num_steps=global_config.nsys_num_steps,
                warmup_steps=global_config.nsys_warmup_steps,
            )
        else:
            raise NotImplementedError(f"Unknown profiler type: {global_config.profiler_type}")
    else:
        profiler = EmptyProfiler()

    model, inputs, optimizer = setup_train_loop(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    run_train_loop(model, inputs, optimizer, profiler)
    if global_config.should_profile and global_config.profiler_type == "torch":
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))