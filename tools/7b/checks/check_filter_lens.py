import argparse
import re
from pathlib import Path

import torch
from common import (
    DEFAULT_CHECKPOINT_NAME,
    EXPLICIT_FILTER_PATTERNS,
    IMPLICIT_FILTER_PATTERN,
    load_state_dict,
    parse_to_list,
)


def check_param_shapes(
    *,
    param_shapes,
    num_groups,
    seq_len,
    target_seq_len,
):
    target_explicit_filter_shape = torch.Size([num_groups, target_seq_len])
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
    seq_len,
    target_seq_len,
):
    target_explicit_filter_shape = torch.Size([num_groups, target_seq_len])
    target_implicit_filter_shape = torch.Size([1, 1, target_seq_len])
    for k in model_dict.keys():
        
        if any(p in k for p in EXPLICIT_FILTER_PATTERNS):
            print(f"Checking {k}...")
            w = model_dict[k]
            assert (
                w.numel() == num_groups * target_seq_len
            ), f"Number of elements mismatch for {k}: {w.numel()} != {num_groups * target_seq_len}"
            assert (
                w.shape[0] == num_groups
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


def check_filter_lens(args, spaces="  "):
    num_groups = args.num_groups
    seq_len = args.seq_len
    target_seq_len = args.target_seq_len
    pat = args.filter_pattern

    # expected_shape = torch.Size([num_groups, seq_len])
    # target_shape = torch.Size([num_groups, target_seq_len])
    print(f"Checking {args.source_dir}")
    # print(f"{spaces}Expected shape: {expected_shape}")
    # print(f"{spaces}Target shape: {target_shape}")
    # print(f"{spaces}Weight pattern: {args.filter_pattern}")

    model_state = load_state_dict(args.source_dir, args.checkpoint_name)
    model_dict = model_state['module']
    param_shapes = model_state['param_shapes']

    check_model_state(
        model_dict=model_dict,
        num_groups=num_groups,
        seq_len=seq_len,
        target_seq_len=target_seq_len,
    )
    check_param_shapes(
        param_shapes=param_shapes,
        num_groups=num_groups,
        seq_len=seq_len,
        target_seq_len=target_seq_len,
    )


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dir", type=str, required=True, help="Source checkpoint directory")
    parser.add_argument("--num_groups", type=int, required=True, help="Number of groups")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    parser.add_argument("--target_seq_len", required=True, type=int, help="Target sequence length")
    parser.add_argument("--filter_pattern", type=parse_to_list, default=EXPLICIT_FILTER_PATTERNS, help="Filter pattern")
    parser.add_argument("--checkpoint_name", type=str, default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument(
        "--optimizer_states", action="store_true", help="Check optimizer states"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    check_filter_lens(args)


# def check_optimizer_states(
#     *,
#     args,
#     model_dict,
#     pat,
#     num_groups,
#     seq_len,
#     target_seq_len,
#     target_shape,
#     spaces="  ",
# ):
#     print("Checking optimizer states...")
#     assert (
#         args.source_dir / "zero"
#     ).exists(), f"Optimizer states directory not found: {args.source_dir}/zero"

#     for k in model_dict.keys():
#         for file in args.source_dir.glob("zero/**/*.pt"):
#             # print(f'Processing {file}')
#             match = re.search(pat, str(file))
#             if match:
#                 state_dict = torch.load(file)
#                 assert "param" in state_dict
#                 param = state_dict["param"]
#                 assert (
#                     param.numel() == num_groups * target_seq_len
#                 ), f"Number of elements mismatch for {k}: {param.numel()} != {num_groups * target_seq_len}"
#                 assert (
#                     param.shape[0] == num_groups
#                     and param.shape[1] == target_seq_len
#                     and param.ndim == 2
#                 ), f"Shape mismatch for {k}: {state_dict[k].shape} != {target_shape}"
#                 assert param[:, seq_len:].sum() == torch.tensor(
#                     0.0, dtype=param.dtype
#                 ), f"Non-zero values found in padding region for {k}"
#                 assert param[:, 0:seq_len].sum() != torch.tensor(
#                     0.0, dtype=param.dtype
#                 ), f"Only zeros found in tensor region for {k}"
#                 print(f"{spaces} -> {'/'.join(str(file).split('/')[-2:])}...passed!")

#     print("Optimizer states check passed!")
