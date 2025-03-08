import argparse

import torch
from common import EXPLICIT_FILTER_PATTERNS, IMPLICIT_FILTER_PATTERN
from partition_lib import get_all_shard_files, load_model_config


def check_param_shapes(
    *,
    param_shapes,
    num_groups,
    model_parallel_size,
    seq_len,
    target_seq_len,
):
    target_explicit_filter_shape = torch.Size([num_groups // model_parallel_size, target_seq_len])
    target_implicit_filter_shape = torch.Size([1, 1, target_seq_len])
    for k in param_shapes.keys():
        if any(p in k for p in EXPLICIT_FILTER_PATTERNS):
            print(f"Checking {k}...")
            assert param_shapes[k] == target_explicit_filter_shape, f"Shape mismatch for {k}: {param_shapes[k]} != {target_explicit_filter_shape}"
        elif IMPLICIT_FILTER_PATTERN in k:
            print(f"Checking {k}...")
            assert param_shapes[k] == target_implicit_filter_shape, f"Shape mismatch for {k}: {param_shapes[k]} != {target_implicit_filter_shape}"
    print("Param shapes check passed!")

def check_model_state(
    *,
    model_dict,
    num_groups,
    model_parallel_size,
    seq_len,
    target_seq_len,
):
    target_explicit_filter_shape = torch.Size([num_groups // model_parallel_size, target_seq_len])
    target_implicit_filter_shape = torch.Size([1, 1, target_seq_len])
    for k in model_dict.keys():
        
        if any(p in k for p in EXPLICIT_FILTER_PATTERNS):
            print(f"Checking {k}...")
            w = model_dict[k]
            assert (
                w.numel() == (num_groups // model_parallel_size )* target_seq_len
            ), f"Number of elements mismatch for {k}: {w.numel()} != {(num_groups // model_parallel_size) * target_seq_len}"
            assert (
                w.shape[0] == num_groups // model_parallel_size
                and w.shape[1] == target_seq_len
                and w.ndim == 2
            ), f"Shape mismatch for {k}: {w.shape} != {target_explicit_filter_shape}"
            assert w[:, seq_len:].sum() == torch.tensor(
                0.0, dtype=w.dtype
            ), f"Non-zero values found in padding region for {k}"
            assert w[:, 0:seq_len].sum() != torch.tensor(
                0.0, dtype=w.dtype
            ), f"Only zeros found in tensor region for {k}"
            print(f"  -> {k}...passed!")
        elif IMPLICIT_FILTER_PATTERN in k:
            print(f"Checking {k}...")
            w = model_dict[k]
            assert w.shape == target_implicit_filter_shape, f"Shape mismatch for {k}: {w.shape} != {target_implicit_filter_shape}"
            print(f"  -> {k}...passed!")
    print("Model weights check passed!")


def check_filter_lens(checkpoint, num_groups, model_parallel_size, seq_len, target_seq_len, model_states_only=False, param_shapes_only=False):
    print(f"Checking {checkpoint}")

    model_state = torch.load(checkpoint, map_location='cpu')
    model_dict = model_state['module']
    param_shapes = model_state['param_shapes']

    if not model_states_only:
        check_model_state(
            model_dict=model_dict,
            num_groups=num_groups,
            model_parallel_size=model_parallel_size,
            seq_len=seq_len,
            target_seq_len=target_seq_len,
        )
    if not param_shapes_only:
        check_param_shapes(
            param_shapes=param_shapes,
            num_groups=num_groups,
            model_parallel_size=model_parallel_size,
            seq_len=seq_len,
            target_seq_len=target_seq_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dir", type=str, required=True, help="Source checkpoint directory")
    parser.add_argument("--source_model_config", type=str, required=True, help="Source model config")
    parser.add_argument("--target_model_config", type=str, required=True, help="Target model config")
    parser.add_argument("--model_states_only", action="store_true", help="Only check model states, not param shapes")
    parser.add_argument("--param_shapes_only", action="store_true", help="Only check param shapes, not model states")
    args = parser.parse_args()

    source_model_config = load_model_config(args.source_model_config)
    target_model_config = load_model_config(args.target_model_config)
    num_groups = target_model_config["num_groups_hyena_medium"]
    model_parallel_size = source_model_config["model_parallel_size"]
    seq_len = source_model_config["seq_length"]
    target_seq_len = target_model_config["seq_length"]

    shard_files = get_all_shard_files(args.source_dir)
    for shard_file in shard_files:
        check_filter_lens(shard_file, num_groups, model_parallel_size, seq_len, target_seq_len, args.model_states_only, args.param_shapes_only)
