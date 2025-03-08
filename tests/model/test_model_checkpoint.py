"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""
import os
import shutil
import torch

import deepspeed
import pytest
import subprocess
from tests.common import (
    distributed_test,
    clear_test_dirs,
    model_setup,
    binary,
    parametrize,
)
import torch

from savanna import initialize_megatron
from savanna.mpu.initialize import initialize_model_parallel
from savanna.arguments import GlobalConfig
from savanna.training import setup_model_and_optimizer


# Pretend like we are on a rank 0 host.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '6040'
os.environ['LOCAL_RANK'] = '0'
torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
initialize_model_parallel(1)
deepspeed.comm.init_distributed(dist_backend='nccl', rank=0, world_size=1)


def test_checkpoint_conversion():
    from savanna.checkpointing import load_checkpoint
    from savanna.checkpointing import save_checkpoint

    args_loaded = GlobalConfig.from_ymls(
        ['tests/test_configs/test_checkpoint1.yml'],
    )
    args_loaded.build_tokenizer()
    initialize_megatron(global_config=args_loaded)
    args_loaded.load = None
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        global_config=args_loaded, use_cache=True
    )

    # save model checkpoint
    save_checkpoint(
        global_config=args_loaded,
        iteration=42,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Shard tensors.
    command = """
    python tools/convert_checkpoint_model_parallel.py \
        --input-checkpoint-dir test_checkpoint/global_step42 \
        --output-checkpoint-dir test_checkpoint_mp4/global_step42 \
        --output-model-parallelism 4
    """
    subprocess.run(command, shell=True)

    # Consolidate tensors.
    command = """
    python tools/convert_checkpoint_model_parallel.py \
        --input-checkpoint-dir test_checkpoint_mp4/global_step42 \
        --output-checkpoint-dir test_checkpoint_mp1/global_step42 \
        --output-model-parallelism 1
    """
    subprocess.run(command, shell=True)

    # reload model from converted checkpoint
    (
        reloaded_model,
        reloaded_optimizer,
        reloaded_lr_scheduler,
        args_reloaded,
    ) = model_setup(['tests/test_configs/test_checkpoint2.yml'], clear_data=False)

    iteration = load_checkpoint(
        global_config=args_reloaded,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        lr_scheduler=reloaded_lr_scheduler,
    )

    # ensure same checkpoint is loaded
    assert (
        iteration == 42
    ), "run_checkpoint_test() iteration loaded from checkpoint correct"

    # check all weight groups are the same
    for idx, ((n1, p1), (n2, p2)) in enumerate(
        zip(
            list(model.module.named_parameters()),
            list(reloaded_model.module.named_parameters()),
        )
    ):
        assert n1 == n2
        params_equal = (p1 == p2).all().item()
        assert params_equal, "run_checkpoint_test() params equal: " + str(n1)


PARAMS_TO_TEST = {
    "pipe_parallel_size,model_parallel_size": [[0, 1], [1, 2], [0, 2], [2, 1]],
    "checkpoint_validation_with_forward_pass": [True],
    "fp16,fp32_allreduce": [
        [
            {
                "enabled": True,
                "type": "bfloat16",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            True,
        ],
        [
            {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            False,
        ],
    ],
}

parameters, names = parametrize(
    PARAMS_TO_TEST, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
)


@pytest.mark.skip
@pytest.mark.parametrize("param_dict", parameters, ids=names)
def test_train(param_dict):
    import tempfile

    d = tempfile.mkdtemp()
    param_dict["save"] = d

    @distributed_test(world_size=2)
    def wrapper():
        run_checkpoint_test(param_dict=param_dict)

    wrapper()


def run_checkpoint_test(yaml_list=None, param_dict=None):
    from savanna.checkpointing import load_checkpoint
    from savanna.checkpointing import save_checkpoint

    model, optimizer, lr_scheduler, args_loaded = model_setup(
        yaml_list, param_dict, clear_data=True
    )

    # save model checkpoint
    save_checkpoint(
        global_config=args_loaded,
        iteration=42,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # reload model from checkpoint
    (
        reloaded_model,
        reloaded_optimizer,
        reloaded_lr_scheduler,
        args_reloaded,
    ) = model_setup(yaml_list, param_dict, clear_data=False)
    iteration = load_checkpoint(
        global_config=args_reloaded,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        lr_scheduler=reloaded_lr_scheduler,
    )

    # ensure same checkpoint is loaded
    assert (
        iteration == 42
    ), "run_checkpoint_test() iteration loaded from checkpoint correct"

    # check all weight groups are the same
    for idx, ((n1, p1), (n2, p2)) in enumerate(
        zip(
            list(model.module.named_parameters()),
            list(reloaded_model.module.named_parameters()),
        )
    ):
        assert n1 == n2
        params_equal = (p1 == p2).all().item()
        assert params_equal, "run_checkpoint_test() params equal: " + str(n1)


if __name__ == "__main__":
    params = list(
        parametrize(
            PARAMS_TO_TEST, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
        )
    )
    test_train(params[0])
