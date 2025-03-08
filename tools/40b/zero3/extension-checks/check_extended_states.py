import argparse
from collections import ChainMap

import torch
from partition_param import get_all_model_files

EXPLICIT_SINGLE_DECAY_BUFFER = "mixer.mixer.filter.decay"
IMPLICIT_MODAL_BUFFER = "mixer.mixer.filter.t"
EXPLICIT_SINGLE_DECAY_PARAM = "mixer.mixer.filter.h"
SEQ_LEN_CHOICES = [128 * 2 ** 10, 256 * 2 ** 10, 512 * 2 ** 10, 1024 * 2 ** 10, 2048 * 2 ** 10]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--mp_size", type=int)
    parser.add_argument("--target_seq_len", type=int, choices=SEQ_LEN_CHOICES)
    parser.add_argument("--num_groups", type=int, default=512)
    
    args = parser.parse_args()

    num_groups = args.num_groups
    mp_size = args.mp_size
    target_seq_len = args.target_seq_len
    checkpoint_dir = args.checkpoint_dir

    assert num_groups % mp_size == 0, f"num_groups {num_groups} must be divisible by mp_size {mp_size}"
    num_groups_per_partition = num_groups // mp_size
    expected_explicit_param_shape = torch.Size([num_groups_per_partition, target_seq_len])
    expected_explicit_buffer_shape = expected_explicit_param_shape
    expected_implicit_buffer_shape = torch.Size([1,1, target_seq_len])

    model_files = get_all_model_files(checkpoint_dir)


    for model_file in model_files:
        print(f"Checking {model_file}")
        model_state = torch.load(model_file, map_location="cpu")
        param_shapes = dict(ChainMap(*model_state['param_shapes']))
        for name, shape in param_shapes.items():
            if EXPLICIT_SINGLE_DECAY_PARAM in name:
                assert shape == expected_explicit_param_shape, f"{model_file} {name} {shape} != {expected_explicit_param_shape}"
    
        for name, param in model_state['module'].items():
            if EXPLICIT_SINGLE_DECAY_BUFFER in name:
                assert param.shape == expected_explicit_buffer_shape, f"{model_file} {name} {param.shape} != {expected_explicit_buffer_shape}"
            elif IMPLICIT_MODAL_BUFFER in name:
                assert param.shape == expected_implicit_buffer_shape, f"{model_file} {name} {param.shape} != {expected_implicit_buffer_shape}"
        print(f"{model_file} passed!")
