import argparse
import re
from pathlib import Path

import torch

#/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-ext/checkpoint_test/202410210618/global_step500000/
# CHKPT_DIR = Path('/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-ext/universal_test/global_step500000_universal/zero')
EXPLICIT_FILTER_PAT = 'mixer.mixer.filter.h'
NUM_GROUPS = 256
SEQ_LEN= 8192
TARGET_SEQ_LEN = 32768

def fix_filter_lens(args):
    num_groups = args.num_groups
    seq_len = args.seq_len
    target_seq_len = args.target_seq_len

    expected_shape = torch.Size([num_groups, seq_len])
    target_shape = torch.Size([num_groups, target_seq_len])
    print(f"Expected shape: {expected_shape}")
    print(f"Target shape: {target_shape}")
    print(f"Weight pattern: {args.pat}")

    # First check model weights
    model_files = list(args.checkpoint_dir.glob('*.pt'))
    assert len(model_files) == 1, f"Expected one model file, found {len(model_files)} {model_files}"
    model_file = model_files[0]
    print(f"Processing model file {model_file}")
    state_dict = torch.load(model_file)
    assert 'module' in state_dict
    model_dict = state_dict['module']

    print("Checking model weights...")
    for k in model_dict.keys():
        if args.pat in k:
            print(f" -> {k} in {model_file}...")
            w = model_dict[k]
            assert w.numel() == num_groups * target_seq_len, f"Number of elements mismatch for {k}: {w.numel()} != {num_groups * target_seq_len}"
            assert w.shape[0] == num_groups and w.shape[1] == target_seq_len and w.ndim == 2, f"Shape mismatch for {k}: {state_dict[k].shape} != {target_shape}"
            assert w[:,seq_len:].sum() == torch.tensor(0.0, dtype=w.dtype), f"Non-zero values found in padding region for {k}"
            assert w[:,0:seq_len].sum() != torch.tensor(0.0, dtype=w.dtype), f"Only zeros found in tensor region for {k}"

    print("Checking optimizer states...")
    for file in args.checkpoint_dir.glob('zero/**/*.pt'):
        # print(f'Processing {file}')
        match = re.search(args.pat, str(file))
        if match:
            print(f' -> {file}...')
            state_dict = torch.load(file)
            for k in state_dict.keys():
                if hasattr(state_dict[k], "shape"):
                    param = state_dict[k]
                    assert param.numel() == num_groups * target_seq_len, f"Number of elements mismatch for {k}: {param.numel()} != {num_groups * target_seq_len}"
                    assert param.shape[0] == num_groups and param.shape[1] == target_seq_len and param.ndim == 2, f"Shape mismatch for {k}: {state_dict[k].shape} != {target_shape}"
                    assert param[:,seq_len:].sum() == torch.tensor(0.0, dtype=param.dtype), f"Non-zero values found in padding region for {k}"
                    assert param[:, 0:seq_len].sum() != torch.tensor(0.0, dtype=param.dtype), f"Only zeros found in tensor region for {k}"
            # for p in file.glob('**/*.pt'):
            #     print(f'Processing {p}')
            #     state_dict = torch.load(p)
            #     for k in state_dict.keys():
            #         if hasattr(state_dict[k], 'shape'):
            #             print(f"Processing {k}: {state_dict[k].shape}")
            #             w = state_dict[k]
            #             assert w.shape == expected_shape
            #             new_w = torch.zeros(target_shape, dtype=w.dtype)
            #             new_w[:, :SEQ_LEN] = w
            #             assert new_w.shape == target_shape
            #             assert new_w[:, :seq_len].equal(w)
            #             print(f"New shape: {new_w.shape}")
            #             state_dict[k] = new_w
            #             print(f"Saving to {p}")
            #             torch.save(state_dict, str(p))
            #         else:
            #             print(f"No params found for {k}: {state_dict[k]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', type=Path)
    parser.add_argument('--pat', type=str, default=EXPLICIT_FILTER_PAT)
    parser.add_argument('--num_groups', type=int, default=NUM_GROUPS)
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN)
    parser.add_argument('--target_seq_len', type=int, default=TARGET_SEQ_LEN)
    args = parser.parse_args()
    fix_filter_lens(args)
