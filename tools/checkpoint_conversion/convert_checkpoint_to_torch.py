import glob
import itertools
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from conversion_utils import (
    EmbeddingPipe,
    Lambda,
    NormPipe,
    ParallelBlockPipe,
    check_buffers,
    convert_module_state_dict,
    detect_module_cls,
    get_extra_states,
    load_ds_checkpoint,
    load_tokenizer_and_model,
    remap_state_dict,
)
from module_mappings import get_savanna_to_vortex_map

SEQUENTIAL_PATTERN = re.compile(r"sequential\.\d+\.")

def check_state_dict(expected, actual, verbose=False):
    # Check if the keys match
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())

    extra_keys = set(actual_keys) - set(expected_keys)
    missing_keys = set(expected_keys) - set(actual_keys)
    common_keys = set(expected_keys) & set(actual_keys)
    if verbose:
        print(f"Actual keys: {len(actual_keys)}, Expected keys: {len(expected_keys)}")
        print(f"Extra keys: {len(extra_keys)}, Missing keys: {len(missing_keys)}")
        print(f"Common keys: {len(common_keys)}")

    assert len(extra_keys) == 0, f"Extra keys found: {extra_keys}"
    assert len(missing_keys) == 0, f"Missing keys found: {missing_keys}"


def remap_model(model_state_dict):
    new_sd = OrderedDict
    for name, module in model_state.named_modules():
        print(f"Name: {name}, Module: {module}")
        breakpoint()
        converted_sd = convert_module_state_dict(state_dict, module=module)

        if isinstance(module, ParallelBlockPipe):
            for k, v in converted_sd.items():
                new_sd[f"blocks.{layer_counter}.{k}"] = v
            layer_counter += 1
        elif isinstance(module, EmbeddingPipe):
            for k, v in converted_sd.items():
                new_sd[f"{k}"] = v
                new_sd["unembed.weight"] = v
        elif isinstance(module, NormPipe):
            for k, v in converted_sd.items():
                new_sd[f"{k}"] = v
    return new_sd

def get_operator_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    operator_config = config['operator-config']
    operators = []
    for operator_type, num_layers in operator_config:
        operators.append(*(operator_type * num_layers))
#    operators = list(itertools.chain.from_iterable(operators))
    assert config['num_layers'] == len(operators), f"Num layers mismatch: {config['num_layers']} != {len(operators)}" 
    return operators

@dataclass
class ModuleParamMappings:
    layer: int
    module_name: str
    module_type: str
    param_names: list[str]
    operator_type: str
    params: dict[str, torch.Tensor]

    def __str__(self):
        return f"Layer: {self.layer}, Module: {self.module_name}, Type: {self.module_type}, Params: {self.param_names}, Operator: {self.operator_type}"
    
    def check(self):
        assert len(self.param_names) == len(self.params), f"Length mismatch: {len(self.param_names)} != {len(self.params)}"

@dataclass
class ModuleOperatorMap:
    type: str
#    layer: int
    params: dict[str, torch.Tensor]

    @property
    def param_names(self):
        return list(self.params.keys())
    
    def __str__(self):
        formatted_params = "\n - ".join(self.param_names)
        return f"{self.type}\n - {formatted_params}"
    
def add_operators(operator_config, state_dict, module_operator_mappings: List[ModuleOperatorMap], prefix=SEQUENTIAL_PATTERN):

    # Start at 2 since the first layer is the embedding layer, second is a lambda layer
    for layer, operator_type in enumerate(operator_config, start=2):
        pat = f"sequential.{layer}."
        layer_params = {k.replace(pat,""): v for k, v in state_dict.items() if k.startswith(pat)}
        operator_mapping = ModuleOperatorMap(type=operator_type, params=layer_params)
        module_operator_mappings.append(operator_mapping)
    return module_operator_mappings

def add_embeddings(state_dict, module_operator_mappings: List[ModuleOperatorMap], embedding_pattern: str = "word_embeddings.weight", embedding_operator_name: str = "embedding"):
    key = [k for k in state_dict if k.endswith(embedding_pattern)]
    assert len(key) == 1, f"Found {len(key)} keys for embedding: {key}"
    embedding_key = key[0]
    embedding_mod = ModuleOperatorMap(type=embedding_operator_name, params={embedding_pattern: state_dict[embedding_key]})
    module_operator_mappings.insert(0, embedding_mod)
    return module_operator_mappings

def add_norms(state_dict, module_operator_mappings: List[ModuleOperatorMap], norm_pattern: str = ".norm.weight", norm_operator_name: str = "norm"):
    keys = [k for k in state_dict if k.endswith(norm_pattern)]
    assert len(keys) == 1, f"Found {len(keys)} keys for norm: {keys}"

    original_key = keys[0]
    # Remove the leading dot
    new_key = norm_pattern[1:]
    norm_mod = ModuleOperatorMap(type=norm_operator_name, params={new_key: state_dict[original_key]})
    module_operator_mappings.append(norm_mod)
    return module_operator_mappings

def check_module_mappings(checkpoint_state_dict, module_operator_mappings: List[ModuleOperatorMap], prefix=SEQUENTIAL_PATTERN):
    all_pipe_params = list(itertools.chain.from_iterable(mapping.param_names for mapping in module_operator_mappings))
    cleaned_checkpoint_keys = [re.sub(prefix, "", k) for k in checkpoint_state_dict.keys()]
    missing_params = set(cleaned_checkpoint_keys) - set(all_pipe_params)
    assert len(missing_params) == 0, f"Missing params: {missing_params}, Num pipe params: {len(all_pipe_params)} Num checkpoint params: {len(checkpoint_state_dict.keys())}"
    return True

def map_modules(checkpoint_state_dict, operator_config):
    module_operator_mappings = []
    module_operator_mappings = add_embeddings(checkpoint_state_dict, module_operator_mappings)
    module_operator_mappings = add_operators(operator_config, checkpoint_state_dict, module_operator_mappings)
    module_operator_mappings = add_norms(checkpoint_state_dict, module_operator_mappings)
    return module_operator_mappings

if __name__ == "__main__":
    
    checkpoint_dir = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/"    
    iteration = 500_000
    tag = f"global_step{iteration}"
    new_checkpoint_path = "./converted_checkpoints/7b"

    # Only used for operator-config
    config_path = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/configs/test/7b_stripedhyena2_base_4M.yml"
    
    model_state_files, optim_state_files = load_ds_checkpoint(checkpoint_dir, tag=tag)
    print(
        f"Found {len(model_state_files)} model state files and {len(optim_state_files)} optimizer state files"
    )
    rank0_model_state = torch.load(model_state_files[0])
    assert len(optim_state_files) == rank0_model_state["dp_world_size"]

    buffer_names = rank0_model_state["buffer_names"]
    checkpoint_state_dict = rank0_model_state["module"]
    operator_config = get_operator_config(config_path)
    print(operator_config)
    module_operator_mappings = map_modules(checkpoint_state_dict, operator_config)

    for i, m in enumerate(module_operator_mappings):
        print(f"Layer{i}: {m}")

    # Checks
    check_module_mappings(checkpoint_state_dict, module_operator_mappings)
    # operators + embeddings + norm
    assert len(module_operator_mappings) == len(operator_config) + 2

    savanna_to_vortex: dict = get_savanna_to_vortex_map()
    for i, mapping in enumerate(module_operator_mappings):
        print(f"Layer{i}: {mapping.type}")
        operator_type = "hyena" if "hyena" in mapping.type else mapping.type
        if not operator_type in savanna_to_vortex:
            print(f"Layer{i}: {mapping.type} not found in savanna_to_vortex")
        param_name_map = savanna_to_vortex[operator_type]
        for k in mapping.param_names:
            new_key = param_name_map.get(k, k)
            print(f" {k} -> {new_key}")
    
    # buffer_check = check_buffers(model_state_dict, buffer_names)
    # print(f"Buffer check: {buffer_check}")
    # tokenizer, model = load_tokenizer_and_model(config_path)
    # keys = list(model.state_dict().keys())

    # model_state_dict = model.state_dict()
    # check_state_dict(model_state_dict, checkpoint_state_dict, verbose=True)

    # sequential_model = model.sequential

    # module_param_mappings = []

    # breakpoint()
    # for i, (name, module) in enumerate(sequential_model.named_children()):
    #     module_type = type(module)
    #     name = module_type.__name__
    #     param_names = [k for k, _ in module.named_parameters()]
    #     fully_qualified_names = [f"sequential.{i}.{k}" for k in param_names]
    #     params = {name: checkpoint_state_dict[name] for name in fully_qualified_names}

    #     operator_type = detect_module_cls(module)
    #     mapping = ModuleParamMappings(
    #         layer=i,
    #         module_name=name,
    #         module_type=module_type,
    #         param_names=fully_qualified_names,
    #         operator_type=operator_type,
    #         params=params
    #     )
    #     module_param_mappings.append(mapping)

    # for mapping in module_param_mappings:
    #     print(mapping)
    #     mapping.check()
    # for key in checkpoint_state_dict.keys():
    #     print(f"Key: {key}")
#    module_param_mappings = {name: [k for k,_ in module.named_parameters()] for name, module in model.named_modules()}
#   for name, param_names in module_param_mappings.items():
# print(f"Name: {name}, Param names: {param_names}")


#     print(f"Name: {name}, Module: {module}")
#     breakpoint()
# new_sd = remap_state_dict(checkpoint_state_dict, )
