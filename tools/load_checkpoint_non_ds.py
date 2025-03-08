# import torch
from savanna.model import BackbonePipe
from savanna import print_rank_0, mpu
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.checkpointing import load_checkpoint
from savanna.training import setup_model_and_optimizer
from savanna.tokenizer import build_tokenizer
import deepspeed
import yaml
import os
import torch


def init_model(global_config, use_cache=False):

    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    return model


def replace_hyphens_with_underscores(data):
    if isinstance(data, dict):
        return {key.replace("-", "_"): replace_hyphens_with_underscores(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_hyphens_with_underscores(element) for element in data]
    else:
        return data


def open_config(config_path):
    with open(config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        # Preprocess the configuration to replace hyphens with underscores
        yaml_config = replace_hyphens_with_underscores(yaml_config)

    return yaml_config


def load_model(config_path, checkpoint_path=None, iteration=None):

    yaml_config = open_config(config_path)

    global_config = GlobalConfig(**yaml_config)

    # make sure checkpoint path and iteration is either passed in or in the config
    if iteration is not None and checkpoint_path is not None:
        pass
    else:
        assert hasattr(global_config, "load") and hasattr(
            global_config, "iteration"
        ), "Need to pass in checkpoint path and iteration!"

        checkpoint_path = global_config.load
        iteration = global_config.iteration

    # optional
    tokenizer = build_tokenizer(global_config)

    torch.set_default_dtype(global_config.params_dtype)

    global_config.rank = 0
    device = "cuda:0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"

    device_num = None
    if ":" in device:
        try:
            device_num = int(device.split(":")[-1])
        except ValueError:
            pass
    os.environ["MASTER_PORT"] = str(7200 + (device_num if device_num is not None else 0))

    initialize_megatron(global_config)

    # get model
    print("Initializing model...")
    model = init_model(global_config).to_sequential()
    checkpoint_iter_path = os.path.join(checkpoint_path, "global_step" + str(iteration))

    for layer_idx in range(len(model.sequential)):
        try:
            layer_path = f"{checkpoint_iter_path}/layer_{layer_idx:02d}-model_00-model_states.pt"
            ckpt = torch.load(layer_path)
            model.sequential[layer_idx].load_state_dict(ckpt)
            print("loaded layer", layer_idx)
        except:
            # note some layers don't have weights.  eg layer_01, and layers -1 and -3 in the list
            print("--did not load layer", layer_idx)

    return model, tokenizer


if __name__ == "__main__":

    """

    Script to load a checkpoint from savanna without relying on DeepSpeed initialization and loading (more manual).

    Usage:
        python load_checkpoint_non_ds.py

    sample config here.
        /home/etnguyen/savanna/configs/evo2/loading_test/1b_10h_6m_6s_2a_50k_1m.yml

    Note:

    - You need to either pass in the `iteration` number and `load` path in the load_model() function, or it will look for it in the config.

    - If passing in the config, pass it in the `iteration` and `load` attributes, just like if you were to resume training.

    TODO:
    - currently doesn't support any model parallelism, we can add that later, just need to change how we parse the .pt files


    """

    # example 1: 1B model
    config_path = "/home/etnguyen/savanna/configs/evo2/loading_test/1b_10h_6m_6s_2a_50k_1m.yml"
    model, tokenizer = load_model(
        config_path,
        checkpoint_path="/checkpoint/hielab/etnguyen/checkpoints/evo2/1b_10h_6m_6s_2a_50k_1m",
        iteration=30000,
    )

    # # example 2:  7B model
    # config_path = "/home/etnguyen/savanna/configs/evo2/7b/8_5/7b_15h_7m_7sc_3a_400k_4node_sc7_fp8.yml"
    # model, tokenizer = load_model(config_path,
    #                    checkpoint_path="/checkpoint/hielab/etnguyen/checkpoints/evo2/7b_15h_7m_7sc_3a_400k_4node_sc7_fp8",
    #                    iteration=2500,
    #                    )

    breakpoint()
