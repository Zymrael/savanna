# import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from savanna.model import BackbonePipe
from savanna import print_rank_0, mpu
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.checkpointing import load_checkpoint
from savanna.training import setup_model_and_optimizer
import deepspeed
import torch
from savanna.tokenizer import build_tokenizer
import yaml


def init_model(global_config, use_cache=False):

    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    return model.to_sequential()


def replace_hyphens_with_underscores(data):
    if isinstance(data, dict):
        return {key.replace("-", "_"): replace_hyphens_with_underscores(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_hyphens_with_underscores(element) for element in data]
    else:
        return data


def get_global_config(config_path):
    with open(config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        # Preprocess the configuration to replace hyphens with underscores
        yaml_config = replace_hyphens_with_underscores(yaml_config)

    return GlobalConfig(**yaml_config)


def set_config_and_env_vars(global_config):
    """
    Setting config and environment variables for deepspeed to work.

    """

    global_config.no_load_optim = True
    global_config.rank = 0
    global_config.launcher = "pdsh"
    global_config.hostfile = None
    global_config.deepspeed_slurm = False
    global_config.include = "localhost@localhost:0"
    global_config.num_gpus = 1
    global_config.gradient_accumulation_steps = 1
    global_config.use_checkpoint_lr_scheduler = False

    device = "cuda:0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    torch.set_default_dtype(global_config.params_dtype)

    device_num = None
    if ":" in device:
        try:
            device_num = int(device.split(":")[-1])
        except ValueError:
            pass
    os.environ["MASTER_PORT"] = str(7200 + (device_num if device_num is not None else 0))


def transfer_layers(model_pipe, model_torch):

    state_dict_model_torch = model_torch.state_dict()

    # loop through model_pipe modules, and find the corresponding layer in model_torch, and set them
    for key, layer in model_pipe.state_dict().items():
        # keys in model_pipe have an extra 'module.' prefix, remove prefix
        key_model_torch = key.removeprefix("module.")
        state_dict_model_torch[key_model_torch] = layer

    # load the state_dict into model_torch
    model_torch.load_state_dict(state_dict_model_torch)

    return model_torch


def load_savanna_checkpoint(config_path, checkpoint_path=None, iteration=None):

    global_config = get_global_config(config_path)

    # make sure checkpoint path and iteration is either passed in or in the config
    if iteration is not None and checkpoint_path is not None:
        # we need to set these for deepspeed to work
        global_config.load = checkpoint_path
        global_config.iteration = iteration
    else:
        assert hasattr(global_config, "load") and hasattr(
            global_config, "iteration"
        ), "Need to pass in checkpoint path and iteration!"

        checkpoint_path = global_config.load
        iteration = global_config.iteration

    set_config_and_env_vars(global_config)

    tokenizer = build_tokenizer(global_config)  # we need to build this

    # initialize megatron
    print("initializing megatron...")
    initialize_megatron(global_config)

    # get model
    print("Initializing model...")
    model_torch = init_model(global_config)

    # init deepspeed
    print("initializing deepspeed...")
    model_pipe, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model_torch,
        optimizer=None,
        args=global_config,
        lr_scheduler=None,
        dist_init_required=False,
        model_parameters=None,
        mpu=mpu if not global_config.is_pipe_parallel else None,
    )

    print("loading checkpoint weights...")
    # Model is loaded in place. (returns iteration, but not needed)
    _ = load_checkpoint(
        global_config=global_config,
        model=model_pipe,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        iteration=global_config.iteration,
    )

    # transfer layers in model_pipe to the plain torch model
    final_model = transfer_layers(model_pipe, model_torch)

    return final_model, tokenizer


if __name__ == "__main__":

    """

    Script to load a checkpoint in raw deepspeed format (eg from Savanna).

    example usage:

    python tools/load_checkpoint_from_deepspeed_raw.py

    - you can pass in the checkpoint path and iteration in the config or in the load_savanna_checkpoint call,
    if in the config, pass the path as "load" arg (just like resume training.)


    """

    # # example 1, small untrained model
    # config_path = "/old_home/etnguyen/savanna/configs/model/evo2/ablations/og2_v2/14l_768d/100m_6h_3m_3s_2a_10k_524k.yml"

    # # # option to either pass here or in config
    # checkpoint_path = "/checkpoint/hielab/etnguyen/checkpoints/evo2/og2_v2/100m_6h_3m_3s_2a_10k_524k_test"
    # iteration = 4

    # model, tokenizer = load_savanna_checkpoint(
    #     config_path,
    #     checkpoint_path=None,
    #     iteration=None,
    # )

    # example 2, trained model
    config_path = "/old_home/etnguyen/savanna/configs/model/evo2/ablations/og2_v2/32l_4096d/7b_13h_8m_8s_3a_cascade15.yml"

    # option to either pass here or in config
    checkpoint_path = "/checkpoint/hielab/brianhie/evo2/7b_13h_8m_8s_3a_cascade15"
    iteration = 110_000

    model, tokenizer = load_savanna_checkpoint(
        config_path,
        checkpoint_path=checkpoint_path,
        iteration=iteration,
    )

    breakpoint()
