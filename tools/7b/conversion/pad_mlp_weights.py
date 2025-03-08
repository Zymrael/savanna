import argparse
import os
from typing import Tuple

import torch
from common import FP8_SHAPE, FP8_SHAPE_T, get_all_shard_files, is_mlp_weight


def pad_to_multiple(d: int, multiple: int) -> int:
    remainder = d % multiple
    if remainder == 0:
        return d, 0
    padding = (multiple - remainder)
    return d+padding, padding

def pad_to_multiple_shape(shape: Tuple[int, int], multiples: Tuple[int, int] = FP8_SHAPE, dim: Tuple[int, int] = (0, 1)) -> Tuple[int, int]:
    assert dim in [(0, 1), (1, 0)], f"Expected dim to be 0 or 1, found {dim}"
    _, bottom_padding = pad_to_multiple(shape[dim[0]], multiples[0])
    _, right_padding = pad_to_multiple(shape[dim[1]], multiples[1])
    return torch.Size((shape[0] + bottom_padding, shape[1] + right_padding))

def pad_to_multiple_tensor(t: torch.Tensor, multiples: Tuple[int, int] = FP8_SHAPE, dim: Tuple[int, int] = (0, 1)) -> torch.Tensor:
    assert t.dim() == 2, f"Expected tensor to have 2 dimensions, found {t.dim()} dimensions"
    assert dim in [(0, 1), (1, 0)], f"Expected dim to be 0 or 1, found {dim}"
    _, bottom_padding = pad_to_multiple(t.size(dim[0]), multiples[0])
    _, right_padding = pad_to_multiple(t.size(dim[1]), multiples[1])
    padding = (0, right_padding, 0, bottom_padding)
    return torch.nn.functional.pad(t, pad=padding, mode="constant", value=0)

def update_param_shapes(param_shapes: dict):
    for param_name in param_shapes.keys():
        if is_mlp_weight(param_name):
            if "w1" in param_name or "w2" in param_name:
                new_shape = pad_to_multiple_shape(param_shapes[param_name], multiples=FP8_SHAPE_T)
                if param_shapes[param_name][0] % FP8_SHAPE_T[0] != 0:
                    print(f"{os.getpid()}: Padding dim 0 of {param_name}: {param_shapes[param_name][0]} -> {new_shape[0]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param_shapes[param_name][0]} -> {new_shape[0]}", flush=True)
                if param_shapes[param_name][1] % FP8_SHAPE_T[1] != 0:
                    print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param_shapes[param_name][1]} -> {new_shape[1]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param_shapes[param_name][1]} -> {new_shape[1]}", flush=True)
                assert new_shape[0] % FP8_SHAPE_T[0] == 0, f"Expected dim 0 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE_T[0]} after padding, found {new_shape[0] % FP8_SHAPE_T[0]}"
                assert new_shape[1] % FP8_SHAPE_T[1] == 0, f"Expected dim 1 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE_T[1]} after padding, found {new_shape[1] % FP8_SHAPE_T[1]}"

                param_shapes[param_name] = new_shape
            elif "w3" in param_name:
                new_shape = pad_to_multiple_shape(param_shapes[param_name], multiples=FP8_SHAPE)
                if param_shapes[param_name][0] % FP8_SHAPE[0] != 0:
                    print(f"{os.getpid()}: Padding dim 0 of {param_name}: {param_shapes[param_name][0]} -> {new_shape[0]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param_shapes[param_name][0]} -> {new_shape[0]}", flush=True)
                if param_shapes[param_name][1] % FP8_SHAPE[1] != 0:
                    print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param_shapes[param_name][1]} -> {new_shape[1]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param_shapes[param_name][1]} -> {new_shape[1]}", flush=True)
                assert new_shape[0] % FP8_SHAPE[0] == 0, f"Expected dim 0 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE[0]} after padding, found {new_shape[0] % FP8_SHAPE[0]}"
                assert new_shape[1] % FP8_SHAPE[1] == 0, f"Expected dim 1 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE[1]} after padding, found {new_shape[1] % FP8_SHAPE[1]}"

                param_shapes[param_name] = new_shape
    return param_shapes

def update_params(model_state: dict):

    param_names = model_state.keys()

    for param_name in param_names:
        param = model_state[param_name]
        if is_mlp_weight(param_name):
            if "w1" in param_name or "w2" in param_name:
                padded_param = pad_to_multiple_tensor(param, multiples=FP8_SHAPE_T)
                assert padded_param[param.shape[0]:, :].equal(torch.zeros_like(padded_param[param.shape[0]:, :]))
                assert padded_param[:, param.shape[1]:].equal(torch.zeros_like(padded_param[:, param.shape[1]:]))
                assert padded_param.shape[0] % FP8_SHAPE_T[0] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE_T} after padding, found {padded_param.shape[0] % FP8_SHAPE_T[0]}"
                assert padded_param.shape[1] % FP8_SHAPE_T[1] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE_T} after padding, found {padded_param.shape[1] % FP8_SHAPE_T[1]}"
                
                if param.shape[0] % FP8_SHAPE_T[0] != 0:
                    print(f"{os.getpid()}: Padding dim 0 of {param_name} from {param.shape[0]} to {padded_param.shape[0]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param.shape[0]} -> {padded_param.shape[0]}", flush=True)
                if param.shape[1] % FP8_SHAPE_T[1] != 0:
                    print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)

                model_state[param_name] = padded_param
            # Pad input dim of w3 to the next multiple of 16
            elif "w3" in param_name:
                padded_param = pad_to_multiple_tensor(param, multiples=FP8_SHAPE)
                assert padded_param[param.shape[0]:, :].equal(torch.zeros_like(padded_param[param.shape[0]:, :]))
                assert padded_param[:, param.shape[1]:].equal(torch.zeros_like(padded_param[:, param.shape[1]:]))
                assert padded_param.shape[0] % FP8_SHAPE[0] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE} after padding, found {padded_param.shape[0] % FP8_SHAPE[0]}"
                assert padded_param.shape[1] % FP8_SHAPE[1] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE} after padding, found {padded_param.shape[1] % FP8_SHAPE[1]}"
                
                if param.shape[0] % FP8_SHAPE[0] != 0:
                    print(f"{os.getpid()}: Padding dim 0 of {param_name}: {param.shape[0]} -> {padded_param.shape[0]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param.shape[0]} -> {padded_param.shape[0]}", flush=True)
                if param.shape[1] % FP8_SHAPE[1] != 0:
                    print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)
                else:
                    print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)

                model_state[param_name] = padded_param

    return model_state

def pad_mlp_weights(checkpoint_dir: str):
    mp_shards = get_all_shard_files(checkpoint_dir)
    assert len(mp_shards) > 0, f"Expected at least one shard file in {checkpoint_dir}"

    for shard_file in mp_shards:
        print(f"Processing {shard_file}", flush=True)
        model_state = torch.load(shard_file, map_location="cpu")
        param_shapes = model_state['param_shapes']
        model_state_dict = model_state['module']
        
        param_shapes = update_param_shapes(param_shapes)
        model_state['param_shapes'] = param_shapes
        
        model_state_dict = update_params(model_state_dict)
        model_state['module'] = model_state_dict
        
        if not args.dry_run:
            print(f"Saving padded model state to {shard_file}", flush=True)
            torch.save(model_state, shard_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pad MLP weights to meet TransformerEngine fp8 requirements.  Updates the checkpoint in place.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Do not save the padded model state to disk")
    args = parser.parse_args()

    pad_mlp_weights(args.checkpoint_dir)
