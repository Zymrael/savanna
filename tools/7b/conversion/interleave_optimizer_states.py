import glob
import re
from collections import defaultdict
from typing import List

import torch
import yaml

MODEL_CONFIG = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/configs/7b-context-ext/model_configs/7b_stripedhyena2_base_4M_32k.yml"
CHECKPOINT_DIR = '/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension/7b_base_4M/universal/interleaved/global_step500000_universal'
DENSE_PROJ_PAT = "mixer.dense_projection.weight"
SHORT_CONV_PROJ_PAT = "mixer.hyena_proj_conv.short_conv_weight"

def filter_files_by_layer(files: List[str], min_layer: int = 2, max_layer: int = 33) -> List[str]:
    """Filters files by their layer number parsed from the filename.

    Args:
        files (List[str]): List of filenames to filter.
        min_layer (int): Minimum layer number (inclusive).
        max_layer (int): Maximum layer number (inclusive).

    Returns:
        List[str]: List of filenames that fall within the specified layer range.
    """
    layers = defaultdict(list)
    for file in files:
        # Extract the layer number by splitting the filename
        match = re.search(r'sequential\.(\d+)\.', file)
        if match:
            try:
                layer_number = int(match.group(1))
                if min_layer <= layer_number <= max_layer:
                    layers[layer_number].append(file)
            except ValueError:
                print(f"Skipping file: {file}")
            
    return layers


def get_flash_layer_nums(operator_config):
        layer_nums = []
        for i, (operator_type, _) in enumerate(operator_config):
            if "flash_v2" in operator_type:
                layer_nums.append(i)

        return layer_nums

def convert_to_interleaved(tensor, dim):
    assert tensor.shape[0] == 3 * dim, f"Expected shape: {3 * dim}, got {tensor.shape[0]}"
    temp_tensor = torch.nn.Parameter(torch.empty_like(tensor))
    temp_tensor[0::3, :] = tensor[:dim, :]
    temp_tensor[1::3, :] = tensor[dim:(2 * dim), :]
    temp_tensor[2::3, :] = tensor[(2 * dim):, :]
    return temp_tensor

def convert_layers(layers, hidden_size):
    print("Converting layers...")
    with torch.no_grad():    
        for layer_num in sorted(layers.keys()):
            files = layers[layer_num]
            for f in files:
                if DENSE_PROJ_PAT in f or SHORT_CONV_PROJ_PAT in f:
                    print(f"   -> Matched layer {f.split('/')[-1]}:{layer_num}")
                    t = torch.load(f)
                    assert 'param' in t
                    param = t['param']
                    shape_before = param.shape
                    t['param'] = convert_to_interleaved(param, dim=hidden_size)
                    shape_after = t['param'].shape
                    assert shape_before == shape_after, f"Shape mismatch: {shape_before} != {shape_after}"
                    print(f"   -> Overwriting original state dict at {f}")
                    torch.save(t, f)    

def remove_flash_layers(layers):
    len_before = len(layers)
    
    for layer_num in flash_layers:
        print(f"Removing flash_v2 layer {layer_num}")
        layers.pop(layer_num)

    len_after = len(layers)
    
    print(f"Removed {len_before - len_after} flash_v2 layers")
    return layers

def get_layers(checkpoint_dir):
    print(f"Getting layers from {checkpoint_dir}")
    optimizer_states = sorted(list(glob.glob(f'{checkpoint_dir}/zero/**/*.pt')))
    layers = filter_files_by_layer(optimizer_states, start_layer, end_layer)
    layer_nums = sorted(list(layers.keys()))
    print(f"Found {len(layers)} layers in checkpoint: {layer_nums}")
    return layers

if __name__ == "__main__":

    model_config = yaml.safe_load(open(MODEL_CONFIG, 'r'))
    hidden_size = model_config['hidden_size']
    num_layers = model_config['num_layers']
    start_layer = 2
    end_layer = start_layer + num_layers - 1

    operator_config = model_config['operator-config']
    layers = get_layers(CHECKPOINT_DIR)    
    assert len(layers) == len(operator_config), f"Number of layers in model config ({len(operator_config)}) does not match the number of layers in the checkpoint ({len(layers)})."
    
    flash_layers = get_flash_layer_nums(operator_config)
    # Need to shift by 2 to match checkpoint layer numbers
    flash_layers = [layer + 2 for layer in flash_layers]
    layers = remove_flash_layers(layers)
    
    print(f"Layers after removing flash_v2 layers: {len(layers)} - {sorted(list(layers.keys()))}")
    
    convert_layers(layers, hidden_size)

    # with torch.no_grad():    
    #     for layer_num in sorted(filtered.keys()):
    #         files = filtered[layer_num]
    #         for f in files:
    #             if DENSE_PROJ_PAT in f or SHORT_CONV_PROJ_PAT in f:
    #                 print(f"Layer {layer_num}: {f}")
    #                 t = torch.load(f)
    #                 assert 'param' in t
    #                 param = t['param']
    #                 print(param.shape)
    #                 t['param'] = reshape_to_interleaved(param, d=hidden_size)
                    
    # layer_types = [op[0][0] for op in operator_config]
    # # update filtered keys with layer types

    # layers = {}
    # for layer_num, layer_type in zip(layer_nums, layer_types):
    #     layers[f"{layer_type}-{layer_num}"] = filtered[layer_num]
    
    # for layer, files in layers.items():
    #     print(f"{layer}: {files}")
    
    # # flash_layers = get_flash_layer_nums(operator_config)
    # # # Need to shift by 2 to match checkpoint layer numbers
    # # flash_layers = [layer + 2 for layer in flash_layers]
    # # # print(f"Flash layers: {flash_layers}")
    # # # for layer_num in flash_layers:
    # # #     layer = filtered[layer_num]
    # # #     print(f"Layer {layer_num}: {layer}")
    # # print(filtered[flash_layers[0]])
    # # print(filtered[flash_layers[0] - 2])    # # print(filtered[flash_layers[0] - 2])