import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).parent.parent
TEST_CONFIGS_DIR = TEST_DIR / "test_configs"

sys.path.append(str(TEST_DIR))
from utils import get_timestamp, get_wandb_run, run_program

from savanna.arguments import GlobalConfig
from savanna.mfu import HyenaFlopsPerIter


def test_model_runner():
    model_config_path = TEST_CONFIGS_DIR / "single_gpu_flash_only.yml"
    model_config = GlobalConfig.from_ymls([model_config_path])

    wandb_project = "test_model_runner"
    wandb_group = "wandb_logging"
    run_name = f"test-{get_timestamp()}"
    run_program(
        model_config_path,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_name=run_name,
    )
    # wait for run to finish and wandb logs to be uploaded, alternatively, can run in offline mode
    time.sleep(10)
    run = get_wandb_run(
        wandb_project=wandb_project, wandb_run_name=run_name, summaries_only=True
    )
    # MFU <= HFU
    # model_flops_per_iteration = flops per iteration = 1 x fwd pass flops + 2x bwd pass flops
    # hw_model_flops_per_iteration = flops per iteration = 1 x fwd pass flops + 2x bwd pass flops + 1x fwd pass if activation_checkpointing
    if not all(
        run["efficiency/model_flops_per_iteration"]
        <= run["efficiency/hw_model_flops_per_iteration"]
    ):
        print(
            f"Test failed - model_flops > hw_model_flops: {run['efficiency/model_flops_per_iteration']} > {run['efficiency/hw_model_flops_per_iteration']}"
        )
    else:
        print(f"Test passed - model_flops <= hw_model_flops: {run['efficiency/model_flops_per_iteration']} <= {run['efficiency/hw_model_flops_per_iteration']}")
    # MFU = (model_flops_per_iteration / num_gpus ) / iteration_time
    # HFU = (hw_model_flops_per_iteration / num_gpus ) / iteration_time
    if not all(run["efficiency/mfu"] <= run["efficiency/hfu"]):
        print(f"Test failed - mfu > hfu: {run['efficiency/mfu']} > {run['efficiency/hfu']}")
    else:
        print(f"Test passed - mfu <= hfu: {run['efficiency/mfu']} <= {run['efficiency/hfu']}")
    expected_keys = [
        "flop_counts",
        "model_flops_per_iteration",
        "hw_model_flops_per_iteration",
        "device_fp16_throughput",
        "theoretical_device_throughput",
    ]
    if model_config.use_fp8_linears:
        expected_keys.append("device_fp8_throughput")

    run_config = run["config"].item()
    for k in expected_keys:
        if k not in run_config:
            print(f"Test failed: {k} not found in config")
        else:
            print(f"Test passed: {k} found in config")
    flop_counts = HyenaFlopsPerIter(**run_config["flop_counts"])
    if model_config.use_fp8_linears:
        expected_device_throughput = int((
            run_config["device_fp8_throughput"] * flop_counts.total_dense_linear_flops
            + run_config["device_fp16_throughput"]
            * (flop_counts.total_flops - flop_counts.total_dense_linear_flops)
        ) / flop_counts.total_flops)
        if expected_device_throughput != run_config["theoretical_device_throughput"]:
            print(
                f"Test failed - expected device throughput != recorded device throughput: {expected_device_throughput} != {run_config['theoretical_device_throughput']}"
            )
        else:
            print(f"Test passed - expected device throughput == recorded device throughput: {expected_device_throughput} == {run_config['theoretical_device_throughput']}")
    else:
        expected_device_throughput = run_config["device_fp16_throughput"]
        if expected_device_throughput != run_config["theoretical_device_throughput"]:
            print(
                f"Test failed - expected device throughput != recorded device throughput: {expected_device_throughput} != {run_config['theoretical_device_throughput']}"
            )
        else: 
            print(f"Test passed - expected device throughput == recorded device throughput: {expected_device_throughput} == {run_config['theoretical_device_throughput']}")

if __name__ == "__main__":
    test_model_runner()