#!/usr/bin/python

import glob
import itertools
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict

import torch


def clean_module(module, to_replace="module.", replace_with=""):
    return {key.replace(to_replace, replace_with): value for key, value in module.items()}

def clean_checkpoints(checkpoint, out_dir, to_replace="module.", replace_with=""):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for checkpoint in checkpoints:
        new_module = {}
        output_path = os.path.join(out_dir, os.path.basename(checkpoint))
        if os.path.exists(output_path):
            print(f"Skipping {checkpoint} as it already exists")
            continue
        module = torch.load(checkpoint)
        cleaned = clean_module(module, to_replace, replace_with)
        new_module['module'] = cleaned
        print(f"Saving cleaned checkpoint to {output_path}")
        torch.save(new_module, output_path)

def check_param_shapes(state_dict: OrderedDict, reference_param_shapes: OrderedDict):
    common_keys = set(state_dict.keys()) & set(reference_param_shapes.keys())
    extra_keys = set(state_dict.keys()) - set(reference_param_shapes.keys())
    missing_keys = set(reference_param_shapes.keys()) - set(state_dict.keys())
    print(f"Extra keys: {extra_keys}")
    print(f"Missing keys: {missing_keys}")

    for key in common_keys:
        assert reference_param_shapes[key] == state_dict[key].shape, f"Expected shape {reference_param_shapes[key]} but got {state_dict[key].shape}"


def check_modules(state_dict: OrderedDict, reference_dict: OrderedDict, reference_param_shapes):
    common_keys = set(state_dict.keys()) & set(reference_dict.keys())
    extra_keys = set(state_dict.keys()) - set(reference_dict.keys())
    missing_keys = set(reference_dict.keys()) - set(state_dict.keys())
    print(f"Extra keys: {len(extra_keys)}")#: \n {extra_keys}")
    print(f"Missing keys: {len(missing_keys)}")#: \n {missing_keys}")
    empty_tensors = []
    nonempty_tensors = []
    for key in common_keys:
        ref_tensor = reference_dict[key]
        state_tensor = state_dict[key]
        assert key in reference_param_shapes
        if len(ref_tensor) == 0:
            print(f"{key} is an empty tensor")
            ref_shape = reference_param_shapes[key]
            actual_shape = state_tensor.shape
            assert ref_shape == actual_shape, f"Expected shape {ref_shape} but got {actual_shape}"
            empty_tensors.append(key)
        else:
            print(f"{key} is a non-empty tensor")
            assert torch.allclose(ref_tensor, state_tensor), f"Tensor mismatch for key {key}"
            nonempty_tensors.append(key)

    return common_keys, extra_keys, missing_keys, empty_tensors, nonempty_tensors
    
def update_module(saved_state_dict, checkpointed_state_dict, inplace=True, verbose=False):
    saved_state_dict = clean_module(saved_state_dict)
    
    if not inplace:
        saved_state_dict = saved_state_dict.copy()
    
    added = []

    for key, value in checkpointed_state_dict.items():
        if key in saved_state_dict:
            if verbose:
                print(f"SKIPPING {key}: Already in saved_state_dict!")
            continue
        if args.skip_extra:
            if verbose:
                print(f"SKIPPING {key}: Not in saved_state_dict!")
            continue
        print(f"ADDING {key} to saved_state_dict")
        if isinstance(value, torch.Tensor):
            assert len(value) > 0, f"Empty tensor for key {key}"
            print(f" -> Added tensor with shape {value.shape} {value.numel()} elements to saved_state_dict")
            saved_state_dict[key] = value
        else:
            print(f" -> Added {type(value)} to saved_state_dict")
            saved_state_dict[key] = value
        added.append(key)

    return saved_state_dict, added

def merge_param_shapes(param_shapes):
    if not isinstance(param_shapes, list):
        return param_shapes
    return OrderedDict(itertools.chain(*[x.items() for x in param_shapes]))

def key_diffs(expected, actual):
    common_keys = set(expected.keys()) & set(actual.keys())
    extra_keys = set(expected.keys()) - set(actual.keys())
    missing_keys = set(actual.keys()) - set(expected.keys())
    return common_keys, extra_keys, missing_keys

def load_extra_state(extra_state):
    extra_state.seek(0)
    extra_state = torch.load(extra_state)
    return extra_state

def create_ds_output_path(i):
    return f"mp_rank_{i:02}_model_states.pt"

if __name__ == "__main__":
    parser = ArgumentParser(description="Clean checkpoints")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing saved state dicts")
    parser.add_argument("--deepspeed_checkpoint", type=str, required=True, help="rank0 zero3 deepspeed checkpoint to use for shape checking")
    parser.add_argument("--global_step", type=int, required=True, help="Global step of the checkpoints")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for cleaned checkpoints")
    parser.add_argument("--dp_world_size", type=int, default=1, help="Data parallel world size")
    parser.add_argument("--source_mp_size", type=int, default=8, help="Source model parallel size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints")
    parser.add_argument("--skip_extra", action="store_true", help="Skip extra keys")
    args = parser.parse_args()

    checkpoints = sorted(list(glob.glob(os.path.join(args.input_dir, "*.pt"))))
    assert len(checkpoints) == args.source_mp_size, f"Num checkpoitns found {len(checkpoints)}, expected {args.source_mp_size}"

    print(f"Found {len(checkpoints)} checkpoints in {args.input_dir}")

    assert os.path.exists(args.deepspeed_checkpoint), f"Deepspeed checkpoint {args.deepspeed_checkpoint} not found"    
    
    ds_checkpoint = torch.load(args.deepspeed_checkpoint)
    print(f"{ds_checkpoint.keys()}")
    ds_state_dict = ds_checkpoint.get('module')
    param_shapes = ds_checkpoint.get('param_shapes')
    param_shapes = merge_param_shapes(param_shapes)
    print(f"Loaded {len(ds_state_dict)} keys from deepspeed checkpoint")
    print(f"Loaded {len(param_shapes)} param shapes from deepspeed checkpoint")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.output_dir, f"updated/global_step{args.step}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    for i, checkpoint in enumerate(checkpoints):

        output_path = os.path.join(output_dir, create_ds_output_path(i))

        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {output_path} as it already exists")
            continue
        print(output_path)

        new_module = {}
        print(f"Cleaning checkpoint {checkpoint}")
        state_dict = torch.load(checkpoint)
        state_dict = clean_module(state_dict)
        common, extra, missing = key_diffs(param_shapes, state_dict)
        print(f"Found {len(common)} common keys, {len(extra)} extra keys, and {len(missing)} missing keys")
        if not (len(state_dict) == len(param_shapes)):
            raise ValueError(f"State dict and param shapes do not match: {len(state_dict)} vs {len(param_shapes)}\n {extra=} {missing=}")
        
        print(f"Updating checkpoint {os.path.basename(checkpoint)}")
        state_dict, new_keys = update_module(state_dict, ds_state_dict)
        print(f"Added {len(new_keys)} new keys to checkpoint")

        new_module['param_shapes'] = param_shapes
        new_module['module'] = state_dict
        new_module['dp_world_size'] = args.dp_world_size

        print(f"Saving updated checkpoint to {output_path}")
        torch.save(new_module, output_path)
    
    end = time.time()
    print(f"Finished in {(end - start):.2f} seconds.  All outputs written to {output_dir}")