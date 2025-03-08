"""
Usage: python convert_checkpoint_model_parallel.py \
           --input-checkpoint-dir /path/to/input/checkpoint \
           --output-checkpoint-dir /path/to/output/checkpoint \
           --output-model-parallelism 2

Loads the (potentially sharded) parameters in `input_checkpoint_dir` and then re-shards
them according to the desired level of model tensor parallelism.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
import os
from pathlib import Path
import torch
from typing import List


@dataclass
class Param:
    name: str       # Name of the parameter in the checkpoint.
    shard_dim: int  # The dimension index that gets sharded. `None` for no sharding.
    hidden_dim: int # The hidden dimension index. `None` for no hidden dimension.


HIDDEN_DIM_SHAPE = None
def concatenate_tensors_across_shards(
        tensor_name: str,
        data_shards: OrderedDict[str, torch.tensor],
        shard_dim: int,
        hidden_dim: int = None,
) -> torch.tensor:
    global HIDDEN_DIM_SHAPE

    tensors = [ shard[tensor_name] for shard in data_shards ]

    if shard_dim is None:
        for tensor in tensors:
            assert torch.allclose(tensors[0], tensor), 'Synchronized params differ.'
        return tensors[0]

    assert shard_dim != hidden_dim, 'No sharding across hidden dimension.'

    # Check that the tensors are the correct shape.
    if hidden_dim is not None:
        for tensor in tensors:
            if HIDDEN_DIM_SHAPE is None:
                HIDDEN_DIM_SHAPE = tensor.shape[hidden_dim]
            assert tensor.shape[hidden_dim] == HIDDEN_DIM_SHAPE, \
                f'Tensor {tensor_name} has invalid hidden shape {tensor.shape}.'

    return torch.cat(tensors, dim=shard_dim)


def split_tensor_across_shards(
        data_shards: List[OrderedDict],
        tensor: torch.tensor,
        tensor_name: str,
        shard_dim: int,
) -> None:
    if shard_dim is None:
        for data_shard in data_shards:
            data_shard[tensor_name] = tensor
        return
    
    n_shards = len(data_shards)
    assert tensor.shape[shard_dim] % n_shards == 0, 'Cannot shard tensor evenly.'

    for chunk, data_shard in zip(
            torch.chunk(tensor, chunks=n_shards, dim=shard_dim),
            data_shards,
    ):
        data_shard[tensor_name] = chunk


def format_layer_filename(layer_name: str, shard: int) -> str:
    return f'{layer_name}-model_{str(shard).zfill(2)}-model_states.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert checkpoint parameters to desired model parallelism."
    )
    parser.add_argument('--input-checkpoint-dir', type=str, required=True,
                        help='Path to the input checkpoint directory')
    parser.add_argument('--output-checkpoint-dir', type=str, required=True,
                        help='Path to the output checkpoint directory')
    parser.add_argument('--output-model-parallelism', type=int, required=True,
                        help='Desired output model parallelism')
    args = parser.parse_args()

    os.makedirs(args.output_checkpoint_dir, exist_ok=True)

    parameter_paths = glob(f'{args.input_checkpoint_dir}/layer_*')

    layer_names = sorted(set([
        path.split('/')[-1].split('-')[0] for path in parameter_paths
    ]))

    detected_input_mp = sum(f'{layer_names[0]}-' in path for path in parameter_paths)
    print(f'Detected input model parallelism of {detected_input_mp}.')

    # List of parameters.
    # Example parameters in the comments (hidden dim 4096).
    param_list = [
        # Only layer_00.
        Param('word_embeddings.weight', 0, 1), #torch.Size([512, 4096])

        Param('input_layernorm.scale', None, 0), #torch.Size([4096])
        Param('post_attention_layernorm.scale', None, 0), #torch.Size([4096])
        Param('pre_mlp_layernorm.scale', None, 0), #torch.Size([4096])
        Param('outer_mlp_layernorm.scale', None, 0), #torch.Size([4096])
        Param('attention.dense_projection.weight', 0, 1), #torch.Size([6144, 4096]),
        Param('attention.dense_projection.bias', 0, None), #torch.Size([6144]),
        Param('attention.hyena_proj_conv.short_conv_weight', 0, None), #torch.Size([6144, 1, 3]),
        Param('attention.hyena_proj_conv.short_conv_bias', 0, None), #torch.Size([6144]),
        Param('attention.mixer.short_conv_weight', 1, None), #torch.Size([3, 2048, 1, 3]),
        Param('attention.mixer.long_conv_bias', 0, None), #torch.Size([2048]),
        Param('attention.mixer.filter.kernel.C', 1, None), #torch.Size([1, 2048, 8, 2]),
        Param('attention.mixer.filter.kernel.log_dt', 0, None), #torch.Size([2048]),
        Param('attention.mixer.filter.kernel.B', 0, None), #torch.Size([2048, 8, 2]),
        Param('attention.mixer.filter.kernel.inv_A_real', 0, None), #torch.Size([2048, 8]),
        Param('attention.mixer.filter.kernel.A_imag', 0, None), #torch.Size([2048, 8]),
        Param('attention.rotary_emb.inv_freq', None, None), #torch.Size([64])
        Param('attention.dense.weight', 1, 0), #torch.Size([4096, 2048]),
        Param('attention.dense.bias', None, 0), #torch.Size([4096])
        Param('mlp.w1.weight', 0, 1), #torch.Size([5464, 4096]),
        Param('mlp.w2.weight', 0, 1), #torch.Size([5464, 4096]),
        Param('mlp.w3.weight', 1, 0), #torch.Size([4096, 5464]),

        # Only last layer.
        Param('norm.scale', None, 0), #torch.Size([4096, 5464]),
    ]
    param_names = set(param.name for param in param_list)

    for layer_name in layer_names:
        print(f'Processing {layer_name}...')
        
        input_layer_paths = sorted([
            path for path in parameter_paths
            if f'{layer_name}-' in path
        ])
        assert len(input_layer_paths) == detected_input_mp, 'Uneven sharding detected.'

        input_data_shards = [ torch.load(path) for path in input_layer_paths ]
        output_data_shards = [ OrderedDict() for _ in range(args.output_model_parallelism) ]

        for data_shard in input_data_shards[1:]:
            assert data_shard.keys() == input_data_shards[0].keys(), \
                'Different parameters found in shards from the same layer.'

        for param_name in input_data_shards[0]:
            assert param_name in param_names, f'Unknown parameter {param_name}.'

        for param in param_list:
            if param.name not in input_data_shards[0]:
                # Parameter not needed for this layer.
                continue
            
            concatenated_tensor = concatenate_tensors_across_shards(
                param.name,
                input_data_shards,
                param.shard_dim,
                param.hidden_dim,
            )
            split_tensor_across_shards(
                output_data_shards,
                concatenated_tensor,
                param.name,
                param.shard_dim,
            )

        for shard, output_data_shard in enumerate(output_data_shards):
            torch.save(
                output_data_shard,
                Path(args.output_checkpoint_dir) / format_layer_filename(layer_name, shard),
            )
