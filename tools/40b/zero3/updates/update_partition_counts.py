import argparse

import torch
from partition_lib import (
    DP_SIZE_KEY,
    MP_SIZE_KEY,
    OPTIMIZER_STATE_KEY,
    PARTITION_COUNT_KEY,
    get_all_model_files,
    get_all_optim_files,
)


def update_partition_counts(checkpoint_dir, partition_count):
    optim_files = get_all_optim_files(checkpoint_dir)
    assert len(optim_files) > 0, f"No optimizer state files found in {checkpoint_dir}"
    for optim_file in optim_files:
        optimizer_state_dict = torch.load(optim_file)
        optimizer_state_dict[OPTIMIZER_STATE_KEY][PARTITION_COUNT_KEY] = partition_count
        print(f"Updated partition count for {optim_file} to {partition_count}")
        torch.save(optimizer_state_dict, optim_file)

def update_world_sizes(checkpoint_dir, target_mp_size, target_dp_size):
    model_files = get_all_model_files(checkpoint_dir)
    for model_file in model_files:
        model_state_dict = torch.load(model_file)
        model_state_dict[DP_SIZE_KEY] = target_dp_size
        model_state_dict[MP_SIZE_KEY] = target_mp_size
        print(f"Updated world sizes for {model_file} mp_world_size: {target_mp_size} dp_world_size: {target_dp_size}")
        torch.save(model_state_dict, model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--partition_count", type=int, required=True)
    parser.add_argument("--target_mp_size", type=int, required=True)
    args = parser.parse_args()
    args.target_dp_size = args.partition_count
    update_partition_counts(args.checkpoint_dir, args.partition_count)
    update_world_sizes(args.checkpoint_dir, args.target_mp_size, args.target_dp_size)
