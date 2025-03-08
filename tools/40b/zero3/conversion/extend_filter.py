import argparse
import os
from pathlib import Path

import torch
from common import (
    EXPLICIT_FILTER_PATTERNS,
    IMPLICIT_FILTER_PATTERN,
    copy_checkpoint_to_new_dir,
)
from einops import rearrange
from partition_lib import get_all_shard_files, load_model_config


def extend_param_shapes(param_shapes, num_groups, model_parallel_size, source_seq_len, target_seq_len):
    expected_explicit_filter_shape = torch.Size([num_groups // model_parallel_size, source_seq_len])
    target_explicit_filter_shape = torch.Size([num_groups // model_parallel_size, target_seq_len])
    expected_implicit_filter_shape = torch.Size([1, 1, source_seq_len])
    target_implicit_filter_shape = torch.Size([1, 1, target_seq_len])

    for k in param_shapes.keys():
        if any(pat in k for pat in EXPLICIT_FILTER_PATTERNS):
            assert param_shapes[k] == expected_explicit_filter_shape
            print(f" -> Fixing {k}, reshaping from {param_shapes[k]} to {target_explicit_filter_shape}...")
            param_shapes[k] = target_explicit_filter_shape
        elif IMPLICIT_FILTER_PATTERN in k:
            assert param_shapes[k] == expected_implicit_filter_shape
            print(f" -> Fixing {k}, reshaping from {param_shapes[k]} to {target_implicit_filter_shape}...")
            param_shapes[k] = target_implicit_filter_shape
        else:
            print(f" -> Skipping {k}, shape {param_shapes[k]}...")
    return param_shapes


def extend_model_state(model_dict, num_groups, model_parallel_size, source_seq_len, target_seq_len):
    """
    Extend the filters in the model state in-place
    """
    expected_explicit_filter_shape = torch.Size([num_groups // model_parallel_size, source_seq_len])
    target_explicit_filter_shape = torch.Size([num_groups // model_parallel_size, target_seq_len])

    expected_implicit_filter_shape = torch.Size([1, 1, source_seq_len])
    target_implicit_filter_shape = torch.Size([1, 1, target_seq_len])

    for k in model_dict.keys():
        if any(pat in k for pat in EXPLICIT_FILTER_PATTERNS):
            print(
                f"   -> Fixing {k}, reshaping from {expected_explicit_filter_shape} to {target_explicit_filter_shape}..."
            )
            w = model_dict[k]
            assert w.shape == expected_explicit_filter_shape
            new_w = torch.zeros(target_explicit_filter_shape, dtype=w.dtype, device=w.device)
            new_w[:, :source_seq_len] = w
            assert new_w.shape == target_explicit_filter_shape
            assert new_w[:, :source_seq_len].equal(w)
            model_dict[k] = new_w
        elif IMPLICIT_FILTER_PATTERN in k:
            print(
                f"   -> Fixing {k}, reshaping from {expected_implicit_filter_shape} to {target_implicit_filter_shape}..."
            )
            w = model_dict[k]
            assert w.shape == expected_implicit_filter_shape
            new_w = rearrange(
                torch.arange(target_seq_len, dtype=torch.float32, device=w.device), "L -> 1 1 L"
            )
            assert new_w.shape == target_implicit_filter_shape
            model_dict[k] = new_w
    return model_dict


def extend_filters(model_state_dict, num_groups, model_parallel_size, source_seq_len, target_seq_len):

    print(f"Extending filters from {source_seq_len} to {target_seq_len}...")

    # First fix model state weights
    print(" Extending model state filters...")
    model_dict = model_state_dict["module"]
    model_dict = extend_model_state(
        model_dict=model_dict,
        num_groups=num_groups,
        model_parallel_size=model_parallel_size,
        source_seq_len=source_seq_len,
        target_seq_len=target_seq_len,
    )
    model_state_dict["module"] = model_dict

    param_shapes = model_state_dict["param_shapes"]
    print(f" -> Extending param shapes...")
    param_shapes = extend_param_shapes(
        param_shapes,
        num_groups=num_groups,
        model_parallel_size=model_parallel_size,
        source_seq_len=source_seq_len,
        target_seq_len=target_seq_len,
    )
    model_state_dict["param_shapes"] = param_shapes

    return model_state_dict

def process_single_shard(num_groups, model_parallel_size, source_seq_len, target_seq_len, source_dir, output_dir, checkpoint_name):
    # Copy source shard to output dir
    source_checkpoint = Path(source_dir) / checkpoint_name
    assert source_checkpoint.exists(), f"Checkpoint {source_checkpoint} does not exist"

    output_path = os.path.join(output_dir, checkpoint_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    copy_checkpoint_to_new_dir(source_checkpoint, output_path)

    # Extend filters
    model_state_dict = torch.load(os.path.join(source_dir, checkpoint_name), map_location='cpu')
    model_state_dict = extend_filters(
        model_state_dict=model_state_dict,
        num_groups=num_groups,
        model_parallel_size=model_parallel_size,
        source_seq_len=source_seq_len,
        target_seq_len=target_seq_len,
    )

    print(f" -> Saving to {output_path}")
    with open(output_path, "wb") as f:
        torch.save(model_state_dict, f)


def main(args):
    source_model_config = load_model_config(args.source_model_config)
    target_model_config = load_model_config(args.target_model_config)

    assert source_model_config["num_groups_hyena_medium"] == target_model_config["num_groups_hyena_medium"]
    
    
    model_parallel_size = source_model_config["model_parallel_size"]
    source_seq_len = source_model_config["seq_length"]
    target_seq_len = target_model_config["seq_length"]
    num_groups = source_model_config["num_groups_hyena_medium"]
    
    source_shard_files = [os.path.basename(shard_file) for shard_file in get_all_shard_files(args.source_dir)]

    for shard_file in source_shard_files:
        process_single_shard(
            num_groups=num_groups,
            model_parallel_size=model_parallel_size,
            source_seq_len=source_seq_len,
            target_seq_len=target_seq_len,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            checkpoint_name=shard_file,
        )

    print(f"Finished processing {len(source_shard_files)} shards")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fix explicit filter lengths in a checkpoint", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--source_model_config", type=str, required=True, help="Path to the source model config"
    )
    parser.add_argument(
        "--target_model_config", type=str, required=True, help="Path to the target model config"
    )

    args = parser.parse_args()

    main(args)
