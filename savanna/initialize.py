"""Megatron initialization."""
import os
import random
import time
import warnings

import deepspeed
import numpy as np
import torch

from savanna import mpu, print_rank_0
from savanna.mpu import set_model_parallel_rank, set_model_parallel_world_size


def initialize_megatron(global_config, allow_no_cuda=False):
    """Set initialize distributed and set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    warnings.filterwarnings("ignore", message="c10d::broadcast_: an autograd kernel was not registered")

    # torch.distributed initialization
    def finish_mpu_init():
        # Pytorch distributed.
        _initialize_distributed(global_config=global_config)

        # Random seeds for reproducibility.
        if global_config.rank == 0:
            print("> setting random seeds to {} ...".format(global_config.seed))
        _set_random_seed(global_config.seed)

    if global_config.lazy_mpu_init:
        global_config.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_model_parallel_world_size(global_config.model_parallel_size)
        # and return function for external DDP manager to call when it has DDP initialized
        set_model_parallel_rank(global_config.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Compile dataset C++ code.
        if global_config.local_rank == 0:
            from savanna.data.data_utils import compile_helper

            compile_helper()

        # Write arguments to tensorboard.
        _write_args_to_tensorboard(global_config=global_config)
        # No continuation function
        return None


def setup_deepspeed_random_and_activation_checkpointing(global_config):
    """Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    """
    num_layers = global_config.num_layers // global_config.checkpoint_num_layers
    num_layers = (
        num_layers if global_config.num_layers % global_config.checkpoint_num_layers == 0 else num_layers + 1
    )

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=global_config.partition_activations,
        contiguous_checkpointing=global_config.contiguous_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=global_config.checkpoint_in_cpu,
        synchronize=global_config.synchronize_each_layer,
        profile=global_config.profile_backward,
    )


def _initialize_distributed(global_config):
    """Initialize torch.distributed and mpu."""
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if global_config.rank == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
        global_config.rank = torch.distributed.get_rank()
        global_config.world_size = torch.distributed.get_world_size()

    else:
        start_time = time.time()
        if global_config.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = global_config.rank % device_count
            if global_config.local_rank is not None:
                assert (
                    global_config.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                global_config.local_rank = device
            torch.cuda.set_device(device)

        deepspeed.init_distributed(
            dist_backend=global_config.distributed_backend,
            auto_mpi_discovery=True,
            distributed_port=os.getenv("MASTER_PORT", "6000"),
            verbose=True,
        )
        global_config.rank = deepspeed.comm.get_rank()
        global_config.world_size = deepspeed.comm.get_world_size()
        
        elapsed_time = time.time() - start_time
        print_rank_0(f"init-distributed ...: {elapsed_time:.2f}")

    # Setup 3D topology.
    pp = global_config.pipe_parallel_size if global_config.pipe_parallel_size >= 1 else 1
    mp = global_config.model_parallel_size if global_config.model_parallel_size >= 1 else 1
    cp = global_config.context_parallel_size if global_config.context_parallel_size >= 1 else 1
    assert (
        global_config.world_size % (pp * mp * cp) == 0
    ), f"world_size={global_config.world_size}, pp={pp}, mp={mp}, cp={cp}"
    dp = global_config.world_size // (pp * mp * cp)

    from deepspeed.runtime.pipe.topology import ProcessTopology

    # this does pipe on the most outside, then data, then context, then model.
    topo = ProcessTopology(axes=["pipe", "data", "context", "model"], dims=[pp, dp, cp, mp])

    # Offset base seeds for the interior pipeline stages.
    # TODO: adjust last stage too once IO is improved.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim("pipe") - 1:
        offset = global_config.seed + 1138
        global_config.seed = offset + (stage_id * mp)

    # Set the model-parallel / data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print(
                "_initialize_distributed() model parallel is already initialized",
                flush=True,
            )
        else:
            mpu.initialize_model_parallel(
                global_config.model_parallel_size,
                global_config.pipe_parallel_size,
                global_config.context_parallel_size,
                dp,
                topology=topo,
                fp32_allreduce=global_config.fp32_allreduce,
            )

    # Init DeepSpeed Activation Checkpointing Features
    setup_deepspeed_random_and_activation_checkpointing(global_config=global_config)


def _init_autoresume(global_config):
    """Set autoresume start time."""

    if global_config.adlr_autoresume:
        print_rank_0("> enabling autoresume ...")
        sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print("> ADLR autoresume is not available, exiting ...", flush=True)
            sys.exit()
        global_config.adlr_autoresume_object = AutoResume

    if global_config.adlr_autoresume_object:
        torch.distributed.barrier()
        global_config.adlr_autoresume_object.init()
        torch.distributed.barrier()


def _set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def _write_args_to_tensorboard(global_config):
    """Write arguments to tensorboard."""
    if global_config.tensorboard_writer:
        for arg_name in vars(global_config):
            global_config.tensorboard_writer.add_text(arg_name, str(getattr(global_config, arg_name)))
