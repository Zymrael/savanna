import argparse

import torch
from extend_zero3_checkpoint import (
    EXPLICIT_SINGLE_DECAY_BUFFER,
    EXPLICIT_SINGLE_DECAY_PARAM,
    IMPLICIT_MODAL_BUFFER,
)
from partition_lib import get_all_model_files, load_model_config


def check_param_shapes(param_shape_groups, model_parallel_size, num_groups, target_seq_len):
    expected_shape = torch.Size([num_groups // model_parallel_size, target_seq_len])


    for param_shape_group in param_shape_groups:
        for name, shape in param_shape_group.items():
            if EXPLICIT_SINGLE_DECAY_PARAM in name:
                assert shape == expected_shape, f"Shape mismatch for {name}: {shape} != {expected_shape}"

def check_buffers(model_dict, model_parallel_size, num_groups, target_seq_len, verbose=False):
    expected_decay_shape = torch.Size([num_groups // model_parallel_size, target_seq_len])
    expected_implicit_modal_shape = torch.Size([1, 1, target_seq_len])

    for name, param in model_dict.items():
        if verbose:
            if hasattr(param, 'shape'):
                print(f"Checking buffer {name} with shape {param.shape}")

        if EXPLICIT_SINGLE_DECAY_BUFFER in name:
            assert param.shape == expected_decay_shape, f"Shape mismatch for {name}: {param.shape} != {expected_decay_shape}"
        elif IMPLICIT_MODAL_BUFFER in name:
            assert param.shape == expected_implicit_modal_shape, f"Shape mismatch for {name}: {param.shape} != {expected_implicit_modal_shape}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--target_model_config", type=str, required=True)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    model_config = load_model_config(args.target_model_config)

    model_parallel_size = model_config["model_parallel_size"]
    num_groups = model_config["num_groups_hyena_medium"]
    target_seq_len = model_config["seq_length"]

    model_files = get_all_model_files(checkpoint_dir)
    assert len(model_files) > 0, f"No model files found in {checkpoint_dir}"

    for model_file in model_files:
        model_state = torch.load(model_file, map_location="cpu")
        param_shape_groups = model_state['param_shapes']

        # check_param_shapes(param_shape_groups, model_parallel_size, num_groups, target_seq_len)
        # print(f"Model {model_file} passed param shape check")
        model_dict = model_state['module']
        check_buffers(model_dict, model_parallel_size, num_groups, target_seq_len, verbose=True)
        print(f"Model {model_file} passed buffer check")
