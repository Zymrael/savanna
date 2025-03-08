"""
Usage: python convert_checkpoint_model_parallel_evo2.py \
           --input-checkpoint-dir /path/to/input/checkpoint/global_step1000 \
           --output-checkpoint-dir /path/to/output/checkpoint_mp2/global_step1000 \
           --output-model-parallelism 2

Loads the (potentially sharded) parameters in `input_checkpoint_dir` and then re-shards
them according to the desired level of model tensor parallelism.

Specialized to the Evo 2 architecture, only supports Zero-1 checkpoints, and does not
convert any optimizer state (only the parameters).
"""
import argparse
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List

import torch
from common import DEFAULT_PARAM_PATTERN
from conversion.params import EVO2_PARAMS, Param

HIDDEN_DIM_SHAPE = None
def concatenate_tensors_across_shards(
        tensor_name: str,
        data_shards: List[OrderedDict[str, torch.Tensor]],
        partition_dim: int,
        hidden_dim: int = None,
        verbose: bool = False
) -> torch.tensor:
    global HIDDEN_DIM_SHAPE

    tensors = [ shard['module'][tensor_name] for shard in data_shards ]

    if partition_dim is None:
        for i, tensor in enumerate(tensors):
            # assert torch.allclose(tensors[0], tensor), f'Synchronized params differ for param {tensor_name}: abs max diff = {(tensors[0] - tensor).abs().max()}.'
            if not torch.allclose(tensors[0], tensor):
                print(f'WARNING: Synchronized params differ for param {tensor_name}: abs max diff = {(tensors[0] - tensor).abs().max()}.')
                # Get the distribution of tensors[0] and tensor
                if verbose:
                    ref_tensor = tensors[0].flatten().to(torch.float32)
                    ref_min, ref_max = ref_tensor.min(), ref_tensor.max()
                    
                    q = torch.tensor([0.25, 0.5, 0.75], device=ref_tensor.device)
                    ref_quantiles = ref_tensor.quantile(q)
                    print(f"rank0 tensor: min={ref_min}, max={ref_max} quantiles={ref_quantiles}")

                    target_tensor = tensor.flatten().to(torch.float32)
                    target_min, target_max = target_tensor.min(), target_tensor.max()
                    target_quantiles = target_tensor.quantile(q)
                    print(f"rank{i} tensor: min={target_min}, max={target_max} quantiles={target_quantiles}")
                    
                    print(f"rank0 tensor distribution:\n {ref_tensor.histc(100, min=ref_min, max=ref_max)}")
                    print(f"rank{i} distribution:\n {target_tensor.histc(100, min=ref_min, max=ref_max)}")

        return tensors[0]

    assert partition_dim != hidden_dim, 'No sharding across hidden dimension.'

    # Check that the tensors are the correct shape.
    if hidden_dim is not None:
        for tensor in tensors:
            if HIDDEN_DIM_SHAPE is None:
                HIDDEN_DIM_SHAPE = tensor.shape[hidden_dim]
            assert tensor.shape[hidden_dim] == HIDDEN_DIM_SHAPE, \
                f'Tensor {tensor_name} has invalid hidden shape {tensor.shape}.'

    return torch.cat(tensors, dim=partition_dim)


def split_tensor_across_shards(
        data_shards: List[OrderedDict],
        tensor: torch.tensor,
        tensor_name: str,
        partition_dim: int,
) -> None:
    if partition_dim is None:
        for data_shard in data_shards:
            data_shard['module'][tensor_name] = tensor
            data_shard['param_shapes'][tensor_name] = tensor.shape
        return
    
    n_shards = len(data_shards)
    assert tensor.shape[partition_dim] % n_shards == 0, 'Cannot shard tensor evenly.'

    for chunk, data_shard in zip(
            torch.chunk(tensor, chunks=n_shards, dim=partition_dim),
            data_shards,
    ):
        data_shard['module'][tensor_name] = chunk.clone()
        data_shard['param_shapes'][tensor_name] = chunk.shape


def format_output_filename(shard: int) -> str:
    return f'mp_rank_{str(shard).zfill(2)}_model_states.pt'


def check_params(detected, expected, param_pattern=DEFAULT_PARAM_PATTERN, verbose=False):

    expected = set(expected) if not isinstance(expected, set) else expected

    model_param_names = []
    for k in detected:
        match = re.search(param_pattern, k)
        if match is not None:
            model_param_names.append(match.group(1))
        else:
            print(f"Could not match {k}")
    detected_param_set = set(model_param_names)

    if verbose:
        print("Detected params in {paths}:\n  {detected_params}".format(paths=parameter_paths[0], detected_params='\n  '.join(detected_param_set)))

    missing_params = expected - detected_param_set
    extra_params = set(p for p in (detected_param_set - expected) if 'extra_state' not in p)
    if len(extra_params) > 0:
        print(f"WARNING: detected extra params: {extra_params}")
    if len(missing_params) > 0:
        print(f"WARNING: missing params: {missing_params}")
    if not (extra_params or missing_params):
        print("No missing or extra params detected!")
    
    # if missing_params or extra_params:
    #     raise ValueError("Missing or extra params detected!")

def convert(input_data_shards, output_data_shards, model_parameter_names: List[str], param_list: List[Param], verbose=False, exclude_extra=False):
    print(f"Converting {len(model_parameter_names)} parameters from {len(input_data_shards)} input shards to {len(output_data_shards)} output shards...")
    converted = 0
    skipped = 0
    for model_parameter in model_parameter_names:
        if args.verbose:
            print(f"Processing {model_parameter}")

        # Ignore FP8 extra state.
        if model_parameter.endswith('._extra_state'):

            if args.verbose:
                print(f'Ignoring {model_parameter} -> contains extra state.')
            skipped += 1
            continue

        # Get the partition dimension and hidden dimension of each parameter.
        param_info = None
        for param in param_list:
            if '.'.join(model_parameter.split('.')[2:]) == param.name:
                assert param_info is None, \
                    f'Found more than one matching parameter for {model_parameter}.'
                param_info = param
        if param_info is None:
            raise ValueError(f'Could not find {model_parameter} among known parameters.')

        concatenated_tensor = concatenate_tensors_across_shards(
            model_parameter,
            input_data_shards,
            param_info.partition_dim,
            param_info.hidden_dim,
            verbose=verbose
        )
        split_tensor_across_shards(
            output_data_shards,
            concatenated_tensor,
            model_parameter,
            param_info.partition_dim,
        )
        converted += 1
    print(f"Converted {converted} of {len(model_parameter_names)} parameters (skipped {skipped} params).")
    num_params = len(output_data_shards[0]['module'])
    print(f"Total params: {num_params}")
    assert all(num_params == len(shard['module']) for shard in output_data_shards), 'Shards have different number of parameters.'

    if not exclude_extra:
        print("Adding extra states from rank0 input shard...")
        rank0_model = input_data_shards[0]['module']
        for k in rank0_model.keys():
            for i, output_shard in enumerate(output_data_shards):
                if k not in output_shard['module']:
                    if i == 0:
                        print(f"Adding {k} to output shards")
                    output_shard['module'][k] = rank0_model[k]
        new_params = len(output_data_shards[0]['module']) - num_params
        print(f"Added {new_params} extra states, total params: {num_params + new_params}")
        assert all(num_params + new_params == len(shard['module']) for shard in output_data_shards), 'Shards have different number of parameters after adding extra states.'

    print("Saving converted checkpoint...")
    for shard, output_data_shard in enumerate(output_data_shards):
        torch.save(
            output_data_shard,
            Path(args.output_dir) / format_output_filename(shard),
        )
    print(f"Converted checkpoint saved to {args.output_dir}")
    
def main(args, param_list: List[Param] = EVO2_PARAMS):

    param_names = set(param.name for param in param_list)

    # Needed when loading from sharded checkpoints since the tensors from mp state files are on different devices
    device = "cuda:0"
    input_data_shards = [ torch.load(path, map_location=device) for path in parameter_paths ]
    output_data_shards = [
        {
            'module': OrderedDict(),
            'param_shapes': OrderedDict(),
            'dp_world_size': input_data_shards[0]['dp_world_size'],
        }
        for _ in range(args.mp_size)
    ]
    model_parameter_names = input_data_shards[0]['module'].keys()
    
    # Check no missing or extra params
    check_params(detected=model_parameter_names, expected=param_names)

    # Convert the checkpoint
    convert(input_data_shards, output_data_shards, model_parameter_names, param_list, verbose=args.verbose, exclude_extra=args.exclude_extra)
    print("Done!")

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint parameters to desired model parallelism.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--source_dir', type=str, required=True, help='Path to the input checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output checkpoint directory')
    parser.add_argument('--mp_size', type=int, required=True, help='Desired output model parallelism')
    parser.add_argument('--exclude-extra', action='store_true', help='Include extra states in the conversion')
    parser.add_argument("--verbose", action="store_true", help="Print more information")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # Pre-checks
    assert os.path.exists(args.source_dir), f'Input checkpoint dir {args.source_dir} not found.'

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Converting checkpoint from {args.source_dir} to {args.output_dir}")

    # Load checkpointed params
    parameter_paths = sorted(glob(f'{args.source_dir}/mp_rank_*_model_states.pt'))
    assert len(parameter_paths) > 0, f'No parameters found in {args.source_dir}'

    main(args)