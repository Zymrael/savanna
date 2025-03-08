import glob

# import torch
import os
import re
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from savanna.model.backbone import EmbeddingPipe, Lambda, NormPipe, ParallelBlockPipe

# import rearrange
# from einops import rearrange
# from opt_einsum import contract


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
import yaml
from module_mappings import MERGE_KEY, PARAMS_TO_MERGE

from savanna import mpu
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.model import BackbonePipe
from savanna.tokenizer import build_tokenizer

# from tools.load_checkpoint_from_deepspeed_raw import load_savanna_checkpoint
_IS_DS_ENV_INIT = False

def remove_state_dict_prefixes(state_dict, patterns=["module.", "sequential."], verbose=False):
    keys = list(state_dict.keys())
    for k in keys:
        new_key = k
        for pattern in patterns:
            new_key = new_key.replace(pattern, "")
        if verbose:
            print(f"Renaming {k} -> {new_key}")
        state_dict[new_key] = state_dict.pop(k)
    return state_dict


def detect_module_cls(module):
    print(f"Module: {module}")
    if isinstance(module, ParallelBlockPipe):
        return module.operator_type
    elif isinstance(module, EmbeddingPipe):
        return "embedding"
    elif isinstance(module, NormPipe):
        return "norm"
    elif isinstance(module, Lambda):
        return "lambda"
    else:
        print(f"Module {module} is not a ParallelBlockPipe, defaulting to attention")
        return "attention"


def get_mapping(operator_type):
    from module_mappings import (
        KEY_UPDATE_DICT_ATTENTION,
        KEY_UPDATE_DICT_EMBEDDING,
        KEY_UPDATE_DICT_HYENA,
        KEY_UPDATE_DICT_NORM,
    )
    
    if operator_type == "hyena":
        mapping = KEY_UPDATE_DICT_HYENA
    elif operator_type == "hyena_medium_conv":
        mapping = KEY_UPDATE_DICT_HYENA
    elif operator_type == "hyena_short_conv":
        mapping = KEY_UPDATE_DICT_HYENA
    elif operator_type == "embedding":
        mapping = KEY_UPDATE_DICT_EMBEDDING
    elif operator_type == "norm":
        mapping = KEY_UPDATE_DICT_NORM
    elif operator_type == "lambda":
        mapping = None
    else:
        mapping = KEY_UPDATE_DICT_ATTENTION

    return mapping

def handle_flash_attention(state_dict, k):

    # # this is to handle the different convention of shapes in FlashAttention module in inference code
    from einops import rearrange

    # rearrange "inner_mha_cls.Wqkv.weight" from [(head * three * dim) x dim] to [three x heads x dim x dim]
    new_Wqkv = rearrange(state_dict[k], '(head three d) d -> head three d d', head=1, three=3)

    # permute 1st and 2nd dims
    new_Wqkv = rearrange(new_Wqkv, 'head three d d -> three head d d', head=1, three=3)

    # recombine the 1st three dims into a single dim
    new_Wqkv = rearrange(new_Wqkv, 'three head d d -> (three head d) d', head=1, three=3)

    # set the new_k value to this new tensor
    state_dict[k] = new_Wqkv
    
    return state_dict


def get_params_to_merge(state_dict, params_to_merge=PARAMS_TO_MERGE):
    to_merge = {}
    for k, v in state_dict.items():
        if k in params_to_merge:
            to_merge[k] = state_dict[k].clone()
    return to_merge

def remap_state_dict(state_dict, module_map, params_to_merge=PARAMS_TO_MERGE, merge_key=MERGE_KEY, include_extra=False):
    new_state_dict = OrderedDict()

    for k in state_dict.keys():
        new_k = module_map.get(k, k)
    
        print(f"Mapping {k} -> {new_k}")

        if new_k != "":
            # Check if the value is a tensor before trying to access .shape #Changed
            if hasattr(state_dict[k], "shape"):
                print(f"{k}: {state_dict[k].shape}")
                if "filter.short_filter_weight" in new_k:
                    new_state_dict[new_k] = state_dict[k][:, None].clone() #state_dict.pop(k)[:, None] #state_dict[k][:, None].clone()
                else:
                    new_state_dict[new_k] = state_dict[k].clone() #state_dict.pop(k) #
            else:
                #Need to handle extra states here
                print(f"{k}: {type(state_dict[k])} (no shape attribute)")
                if include_extra:
                    new_state_dict[new_k] = state_dict[k] #state_dict.pop(k)

    tensors_to_merge = get_params_to_merge(state_dict, module_map, params_to_merge=params_to_merge)
    
    if len(tensors_to_merge) > 0:
        assert not any([k in new_state_dict for k in tensors_to_merge.keys()]), "Merged keys already exist in state dict"
        assert [v.shape for v in tensors_to_merge.values()].count(tensors_to_merge.values()[0].shape) == len(tensors_to_merge), "Tensors to merge have different shapes"
        tensor_key = ", ".join(list(tensors_to_merge.keys()))
        params_to_merge_values = torch.cat(tensors_to_merge.values(), dim=0)
        print(f"Merging {tensor_key} -> {merge_key}")
        new_state_dict[merge_key] = params_to_merge_values

    return new_state_dict


def convert_module_state_dict(state_dict, module, params_to_merge=PARAMS_TO_MERGE, merge_key=MERGE_KEY):
    """
    Convert a pretrained savanna checkpoint state_dict to stripedhyena format

    Note:
    - Keys are replaced according to handrolled mapping
    - QKV weights are concatenated into a single tensor
    - Modal parametrization converted to pole / residue form
    """
    
    operator_type = detect_module_cls(module)
    print(f"Operator type: {operator_type}")
    module_map = get_mapping(operator_type)
    
    # lambdas
    if module_map is None:
        return state_dict

    new_state_dict = OrderedDict()
    
    breakpoint()
    state_dict = remove_state_dict_prefixes(state_dict)

    print(f"State dict keys: {state_dict.keys()}")
    breakpoint()
    new_state_dict = remap_state_dict(state_dict, module_map, params_to_merge=params_to_merge, merge_key=merge_key)
    
    # We extract pole and residue representation for long hyena, medium and short use h instead
    if operator_type == "hyena" or operator_type == "hyena_medium_conv":

        if operator_type == "hyena":
            p = state_dict["mixer.mixer.filter.p"].reshape(4096, 16).to(torch.float32)
            R = state_dict["mixer.mixer.filter.R"].reshape(4096, 16).to(torch.float32)

            new_state_dict.pop("mixer.mixer.filter.p")
            new_state_dict.pop("mixer.mixer.filter.R")

            gamma = state_dict["mixer.mixer.filter.gamma"].to(torch.float32)

            logp = -torch.exp(p)
            logp = (logp * torch.exp(gamma))[..., None]

            new_state_dict["filter.log_poles"] = logp
            new_state_dict["filter.residues"] = R

            # @mp: code for previous parametrizations of long hyena
            # new_state_dict["filter.poles"] = new_state_dict["mixer.mixer.filter.p"].reshape(4096, 8, 1, 2)
            # new_state_dict["filter.residues"] = new_state_dict["mixer.mixer.filter.R"].reshape(4096, 8, 1, 2)

            # new_state_dict.pop("mixer.mixer.filter.p")
            # new_state_dict.pop("mixer.mixer.filter.R")

            # print(f"filter.poles shape: {new_state_dict['filter.poles'].shape}")
            # print(f"filter.residues shape: {new_state_dict['filter.residues'].shape}")

        elif operator_type == "hyena_medium_conv":
            # print(dir(state_dict))
            # parametrization in medium_conv matches! need to slice h with hyena_medium_conv_len
            h = state_dict["mixer.mixer.filter.h"]
            decay = state_dict["mixer.mixer.filter.decay"]
            L = module.mixer.mixer.hyena_medium_conv_len
            print(f"h shape: {h.shape}")
            h = h[:, :L] * decay[:, :L]
            new_state_dict["filter.h"] = h.unsqueeze(1)

    elif operator_type == "hyena_short_conv":
        print(state_dict["mixer.mixer.short_conv.short_conv_weight"].shape)
        h = state_dict["mixer.mixer.short_conv.short_conv_weight"]
        print(f"h shape: {h.shape}")
        new_state_dict["filter.h"] = h

    return new_state_dict


def checkpoint_conversion(checkpoint_path, new_checkpoint_path):
    # loads checkpoint in deepspeed format ("layer-{idx}-model_00-model_states.pt")
    # assumes model parallel 1
    "Deprecated: we use deepspeed raw checkpoint loading to convert the entire model"
    files = glob.glob(os.path.join(checkpoint_path, "layer*states.pt"))
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split("_")[1].split("-")[0]))
    for idx, file in enumerate(files):
        state_dict = torch.load(file)
        print(f"Loading {file}, keys: {state_dict.keys()}", end="\n\n")

        state_dict = convert_module_state_dict(state_dict)

        new_file = file.split("/")[-1]
        torch.save(state_dict, os.path.join(new_checkpoint_path, f"layer_{idx:02d}.pt"))


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

def set_local_deepspeed_env():
    global _IS_DS_ENV_INIT
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["SLURM_NTASKS"] = "1"
    os.environ["SLURM_NTASKS_PER_NODE"] = "1"
    os.environ["SLURM_GPUS_ON_NODE"] = "1"
    os.environ["GLOBAL_NUM_GPUS"] = "1"
    _IS_DS_ENV_INIT = True

def set_config(global_config, single_device=True):
    """
    Setting config and environment variables for deepspeed to work.
    """
    global _IS_DS_ENV_INIT
    if not _IS_DS_ENV_INIT and single_device:
        set_local_deepspeed_env()

    global_config.no_load_optim = True
    global_config.launcher = "pdsh"
    global_config.hostfile = None
    global_config.deepspeed_slurm = False

    global_config.gradient_accumulation_steps = 1
    global_config.use_checkpoint_lr_scheduler = False

    # Single Device
    if single_device:
        global_config.rank = 0
        global_config.include = "localhost@localhost:0"
        global_config.num_gpus = 1

    device = f"cuda:{global_config.rank}"
    torch.set_default_dtype(global_config.params_dtype)

    device_num = None
    if ":" in device:
        try:
            device_num = int(device.split(":")[-1])
        except ValueError:
            pass
    os.environ["MASTER_PORT"] = str(7200 + (device_num if device_num is not None else 0))

def init_model(global_config, use_cache=False):

    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    return model


def get_tokenizer(global_config):
    if not _IS_DS_ENV_INIT:
        set_local_deepspeed_env()
    tokenizer = build_tokenizer(global_config)
    return tokenizer

def load_tokenizer_and_model(config_path):
    
    if not _IS_DS_ENV_INIT:
        print("Setting local deepspeed env...")
        set_local_deepspeed_env()

    global_config = get_global_config(config_path)
    set_config(global_config)

    # Needed to vocab padding when initializing model
    tokenizer = get_tokenizer(global_config)

    # initialize megatron
    print("initializing megatron...")
    initialize_megatron(global_config)

    # get model
    print("Initializing model...")
    model_torch = init_model(global_config).to_sequential()
    return tokenizer, model_torch

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
    model_torch = init_model(global_config).to_sequential()
    return model_torch


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_checkpoint_files(checkpoint_dir, glob_pattern):
    # XXX: need to test that this simple glob rule works for multi-node setup too
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_model_files_by_rank(checkpoint_dir, rank=None):
    rank_pat = f"{rank:02}" if rank is not None else "*"
    return get_checkpoint_files(checkpoint_dir, f"*mp_rank_{rank_pat}_model_states.pt")


def get_optim_files_by_rank(checkpoint_dir, rank=None):
    rank_pat = f"{rank:02}" if rank is not None else "*"
    return get_checkpoint_files(checkpoint_dir, f"*mp_rank_{rank_pat}_optim_states.pt")


def load_ds_checkpoint(checkpoint_dir, tag=None):
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, "latest")
        assert os.path.exists(latest_path), f"latest checkpoint not found at {latest_path}"
        tag = open(latest_path, "r").read().strip()

    print(f"Loading checkpoint with tag: {tag}")
    checkpoint_path = os.path.join(checkpoint_dir, tag)
    model_state_files = get_model_files_by_rank(checkpoint_path)
    optim_state_files = get_optim_files_by_rank(checkpoint_path)
    return model_state_files, optim_state_files

def check_buffers(model, buffer_names, verbose=False):
    if verbose:
        for buffer in buffer_names:
            if buffer not in model:
                print(f"Buffer {buffer} not found in model state")
            else:
                print(f"{buffer}: {model[buffer].shape}")
    return len(set(model.keys()) & set(buffer_names)) == len(buffer_names)


def load_extra_state(extra_state: BytesIO):
    extra_state.seek(0)
    extra_state_dict = torch.load(extra_state)
    return extra_state_dict


def get_extra_states(model, key="extra"):
    extra_states = {}
    for k, v in model.items():
        if key in k:
            assert isinstance(v, BytesIO)
            extra_states[k] = load_extra_state(v)
    return extra_states


def load_savanna_model():

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["SLURM_NTASKS"] = "1"
    os.environ["SLURM_NTASKS_PER_NODE"] = "1"
    os.environ["SLURM_GPUS_ON_NODE"] = "1"
    os.environ["GLOBAL_NUM_GPUS"] = "1"

    breakpoint()
    model, tokenizer = load_savanna_checkpoint(
        config_path,
        checkpoint_path=checkpoint_path,
        iteration=iteration,
    )

def remap_module(model):
    new_state_dict = OrderedDict()
    layer_counter = 0
    
    for name, module in model.named_modules():
        print(f"Mapping {name} of type {type(module)}")

        converted_state_dict = convert_module_state_dict(module.state_dict(), module)

        if isinstance(module, ParallelBlockPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"blocks.{layer_counter}.{k}"] = v
            layer_counter += 1
        elif isinstance(module, EmbeddingPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v
                new_state_dict["unembed.weight"] = v
        elif isinstance(module, NormPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v

    return new_state_dict

def run_validation(model, tokenizer):
    sequential = model.sequential

    device = sequential[0].word_embeddings.weight.device

    inputs = tokenizer.tokenize("ACTGACTGACTGACTG")
    inputs = torch.tensor(inputs)[None].to(device).long()

    outputs = sequential((inputs, None, None))

    print(inputs, inputs.shape)
    print(outputs, outputs.shape)

    os.makedirs(new_checkpoint_path, exist_ok=True)
    torch.save(outputs, os.path.join(new_checkpoint_path, "logits_test.pt"))

    new_state_dict = OrderedDict()
    layer_counter = 0
    for idx, module in enumerate(sequential):
        converted_state_dict = convert_module_state_dict(module.state_dict(), module)

        if isinstance(module, ParallelBlockPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"blocks.{layer_counter}.{k}"] = v
            layer_counter += 1
        elif isinstance(module, EmbeddingPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v
                new_state_dict["unembed.weight"] = v
        elif isinstance(module, NormPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v


def save_checkpoint(new_checkpoint_path, new_state_dict):
    print(new_state_dict.keys())
    checkpoint_file = f"iter_{iteration}.pt"
    torch.save(new_state_dict, os.path.join(new_checkpoint_path, checkpoint_file))
    torch.save(new_state_dict, os.path.join(new_checkpoint_path, checkpoint_file))
