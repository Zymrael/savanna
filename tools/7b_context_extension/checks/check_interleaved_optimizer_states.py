import argparse
import glob
import re
from collections import defaultdict
from typing import List

import torch
import yaml

DEFAULT_MODEL_CONFIG = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/configs/7b-context-ext/model_configs/7b_stripedhyena2_base_4M_32k.yml"
LEGACY_CHECKPOINT_DIR = '/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension/7b_base_4M/universal/global_step500000_universal'

UPDATED_INTERLEAVE_DIR = '/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension/7b_base_4M/universal_interleaved/global_step500000_universal'
UPDATED_CONTEXT_LEN_DIR = UPDATED_INTERLEAVE_DIR.replace("7b_base_4M", "7b_32K")

UPDATED_CHECKPOINT_DIR = UPDATED_INTERLEAVE_DIR

DEFAULT_CHECKPOINT_NAME = "mp_rank_00_model_states.pt"

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

def check_interleaved(original: torch.Tensor, interleaved: torch.Tensor, dim: int):
    # check_sum = (original.sum() == interleaved.sum())
    # passed = False
    # if check_sum:
    #     print(f"   -> Check sum passed: {original.sum():.6f} == {interleaved.sum():.6f}")
    # else:
    #     print(f"   -> Check sum failed: {original.sum():.6f} != {interleaved.sum():.6f}")
    
    passed = False
    for i in range(3):
        start = i * dim
        end = (i + 1) * dim
        expected = original[start:end, :]
        actual = interleaved[i::3, :]
        diff = (expected - actual).abs().max()
        if not torch.equal(expected, actual):
            print(f"   -> Interleaved at idx {i} failed: {diff:.4f}")
            passed = False      
        else:
            print(f"   -> Interleaved at idx {i} passed: {diff:.4f}")
            passed = True
    return passed

def check_layers(original_layers, updated_layers, hidden_size):
    assert original_layers.keys() == updated_layers.keys(), "Layer numbers do not match"
    all_passed = True
    for layer_num in sorted(original_layers.keys()):
        old_files, updated_files = sorted(original_layers[layer_num]), sorted(updated_layers[layer_num])
        assert len(old_files) == len(updated_files), f"Number of files do not match for layer {layer_num}"

        for f1, f2 in zip(old_files, updated_files):
             
            if DENSE_PROJ_PAT in f1 or SHORT_CONV_PROJ_PAT in f1:
                assert DENSE_PROJ_PAT in f2 or SHORT_CONV_PROJ_PAT in f2, f"File names do not match: {f1} - {f2}"

                print(f" Matched layer {f1.split('/')[-1]}:{layer_num}")
                original_sd = torch.load(f1)
                updated_sd = torch.load(f2)
                assert 'param' in original_sd and 'param' in updated_sd, "Param key not found in state dict"

                original_tensor = original_sd['param']
                updated_tensor = updated_sd['param']
                assert original_tensor.shape == updated_tensor.shape, "Tensor shapes do not match"

                passed = check_interleaved(original_tensor, updated_tensor, dim=hidden_size)
                fails = []
                if passed:
                    print(f" -> Checks passed for layer {layer_num}, {f1.split('/')[-1]} {f2.split('/')[-1]}")
                else:
                    all_passed = False
                    fails.append((f1, f2, layer_num))
                    print(f" -> Checks failed for layer {layer_num}, {f1.split('/')[-1]} {f2.split('/')[-1]}")

        if len(fails) > 0:
            print("Failed layers:")
            for f1, f2, layer_num in fails:
                print(f" -> {layer_num}: {f1} - {f2}")

        assert all_passed
    
    print("All checks passed!")

def get_flash_layer_nums(operator_config):
        layer_nums = []
        for i, (operator_type, _) in enumerate(operator_config):
            if "flash_v2" in operator_type:
                layer_nums.append(i)

        return layer_nums

def remove_flash_layers(layers, flash_layers, verbose=False):
    len_before = len(layers)
    
    for layer_num in flash_layers:
        if verbose:
            print(f"Removing flash_v2 layer {layer_num}")
        layers.pop(layer_num)

    len_after = len(layers)
    
    print(f"Removed {len_before - len_after} flash_v2 layers")
    return layers

def check_flash_layers(original_layers, updated_layers, flash_layers):
    assert original_layers.keys() == updated_layers.keys(), "Layer numbers do not match"
    all_passed = True
    for layer_num in sorted(flash_layers):
        old_files, updated_files = sorted(original_layers[layer_num]), sorted(updated_layers[layer_num])
        assert len(old_files) == len(updated_files), f"Number of files do not match for layer {layer_num}"

        for f1, f2 in zip(old_files, updated_files):
             
            if DENSE_PROJ_PAT in f1 or SHORT_CONV_PROJ_PAT in f1:
                assert DENSE_PROJ_PAT in f2 or SHORT_CONV_PROJ_PAT in f2, f"File names do not match: {f1} - {f2}"

                print(f" Matched layer {f1.split('/')[-1]}:{layer_num}")
                original_sd = torch.load(f1)
                updated_sd = torch.load(f2)
                assert 'param' in original_sd and 'param' in updated_sd, "Param key not found in state dict"

                original_tensor = original_sd['param']
                updated_tensor = updated_sd['param']
                assert original_tensor.shape == updated_tensor.shape, "Tensor shapes do not match"

                passed = original_tensor.equal(updated_tensor)
                fails = []
                if passed:
                    print(f" -> Checks passed for layer {layer_num}, {f1.split('/')[-1]} {f2.split('/')[-1]}")
                else:
                    all_passed = False
                    fails.append((f1, f2, layer_num))
                    print(f" -> Checks failed for layer {layer_num}, {f1.split('/')[-1]} {f2.split('/')[-1]}")

        if len(fails) > 0:
            print("Failed layers:")
            for f1, f2, layer_num in fails:
                print(f" -> {layer_num}: {f1} - {f2}")

        assert all_passed
    
    print("All flash layer checks passed!")

def get_layers(checkpoint_dir, start_layer, end_layer):
    optimizer_states = sorted(list(glob.glob(f'{checkpoint_dir}/zero/**/*.pt')))
    layers = filter_files_by_layer(optimizer_states, start_layer, end_layer)
    
    return layers


def get_args():
    parser = argparse.ArgumentParser(description='Convert model checkpoint to interleaved')
    parser.add_argument('--legacy_checkpoint', type=str, default=LEGACY_CHECKPOINT_DIR, help="Path to model checkpoint directory")
    parser.add_argument("--updated_checkpoint", type=str, default=UPDATED_CHECKPOINT_DIR, help="Path to updated model checkpoint")
    parser.add_argument('--checkpoint_name', type=str, default=DEFAULT_CHECKPOINT_NAME, help="Name of model checkpoint to load")
    parser.add_argument('--config_path', type=str, default=DEFAULT_MODEL_CONFIG, help="Path to model config")
    return parser.parse_args()

def main(args):
    print(f"Loading model config from {args.config_path}")
    model_config = yaml.safe_load(open(args.config_path, 'r'))
    hidden_size = model_config['hidden_size']
    num_layers = model_config['num_layers']
    print(f"Hidden size: {hidden_size}, Number of layers: {num_layers}")
    start_layer = 2
    end_layer = start_layer + num_layers - 1

    operator_config = model_config['operator-config']
    original_layers = get_layers(args.legacy_checkpoint, start_layer, end_layer)
    print(f"Found {len(original_layers)} layers in checkpoint {args.legacy_checkpoint}")
    updated_layers = get_layers(args.updated_checkpoint, start_layer, end_layer)    
    print(f"Found {len(updated_layers)} layers in checkpoint {args.updated_checkpoint}")
    
    assert len(original_layers) == len(updated_layers) == len(operator_config), f"Number of layers in model config ({len(operator_config)}) does not match the number of layers in legacy checkpoint ({len(original_layers)}) and updated checkpoint ({len(updated_layers)})."
    
    # Get flash layer numbers
    flash_layers = get_flash_layer_nums(operator_config)
    # Need to shift by 2 to match checkpoint layer numbers
    flash_layers = [layer + 2 for layer in flash_layers]
    print("Checking flash layers...")
    check_flash_layers(original_layers, updated_layers, flash_layers)

    # Remove flash layers and check remaining
    original_layers = remove_flash_layers(original_layers, flash_layers)
    updated_layers = remove_flash_layers(updated_layers, flash_layers)
    
    assert len(original_layers) == len(updated_layers), f"After removing flash_v2 layers, number of layers does not match the number of layers in legacy checkpoint ({len(original_layers)}) and updated checkpoint ({len(updated_layers)})."
    assert original_layers.keys() == updated_layers.keys(), "Layer numbers do not match"

    print(f"Layers after removing flash_v2 layers from original checkpoint: {len(original_layers)} - {sorted(list(original_layers.keys()))}")
    print(f"Layers after removing flash_v2 layers from interleaved checkpoint: {len(updated_layers)} - {sorted(list(updated_layers.keys()))}")
    print("Checking non-flash layers...")
    check_layers(original_layers, updated_layers, hidden_size)

if __name__ == "__main__":
    args = get_args()
    main(args)
        