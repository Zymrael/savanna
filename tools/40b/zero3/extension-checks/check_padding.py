import argparse

import torch
from common import (
    EXPLICIT_SINGLE_DECAY_BUFFER,
    EXPLICIT_SINGLE_DECAY_PARAM,
    IMPLICIT_MODAL_BUFFER,
)
from partition_lib import get_all_model_files, load_model_config


def check_param_shapes(model_state, target_model_config):
    model_parallel_size = target_model_config['model_parallel_size']
    num_groups = target_model_config['num_groups_hyena_medium']
    seq_length = target_model_config['seq_length']

    for param_shape_group in model_state['param_shapes']:
        for name, shape in param_shape_group.items():
            if EXPLICIT_SINGLE_DECAY_PARAM in name:
                assert shape == torch.Size([num_groups // model_parallel_size, seq_length]), f"Explicit single decay param shape mismatch: {shape} != {torch.Size([num_groups // model_parallel_size, seq_length])}"
    
    for name, param in model_state['module'].items():
        if EXPLICIT_SINGLE_DECAY_BUFFER in name:
            assert param.shape == torch.Size([num_groups // model_parallel_size, seq_length]), f"Explicit single decay buffer shape mismatch: {param.shape} != {torch.Size([num_groups // model_parallel_size, seq_length])}"
        elif IMPLICIT_MODAL_BUFFER in name:
            assert param.shape == torch.Size([1, 1, seq_length]), f"Implicit modal buffer shape mismatch: {param.shape} != {torch.Size([1, 1, seq_length])}"
                
def check_padding(model_state):
    failed_count = 0
    for param_shape_group in model_state['param_shapes']:
            for name, shape in param_shape_group.items():
                if "w1" in name or "w2" in name:
                    if shape[0] % 16 != 0 and shape[1] % 8 != 0:
                        print(f"{model_file} failed {name} {shape}", flush=True)
                        failed_count += 1
                elif "w3" in name:
                    if shape[1] % 16 != 0 and shape[0] % 8 != 0:
                        print(f"{model_file} failed {name} {shape}", flush=True)
                        failed_count += 1
                else:
                    print(f"{model_file} passed {name} {shape}", flush=True)
        
    print(f"Total failed: {failed_count}", flush=True)
    assert failed_count == 0, f"Total failed {model_file}: {failed_count}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--target_model_config", type=str)
    args = parser.parse_args()

    model_files = get_all_model_files(args.checkpoint_dir)
    target_model_config = load_model_config(args.target_model_config)

    for model_file in model_files:
        failed_count = 0

        print(f"Checking {model_file}", flush=True)
        model_state = torch.load(model_file, map_location="cpu")

        # Check padding
        check_padding(model_state)
        print(f"Padding check passed for {model_file}", flush=True)

        # Check param shapes
        check_param_shapes(model_state, target_model_config)
        print(f"Param shapes check passed for {model_file}", flush=True)

    print(f"All checks passed!", flush=True)