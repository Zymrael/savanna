import glob
import os
import re
from typing import Tuple

import torch
import yaml

MLP_WEIGHT_PATTERN = r".*mlp.w[0-9]{1}.weight$"
FP8_SHAPE = (8, 16)
FP8_SHAPE_T = (16, 8)
EXPLICIT_FILTER_PATTERNS = ["mixer.mixer.filter.h", "mixer.mixer.filter.decay"]
IMPLICIT_FILTER_PATTERN = "mixer.mixer.filter.t"

def is_mlp_weight(param_name: str) -> bool:
    return re.match(MLP_WEIGHT_PATTERN, param_name) is not None

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_checkpoint_files(checkpoint_dir, glob_pattern):
    # XXX: need to test that this simple glob rule works for multi-node setup too
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_all_model_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, f"*model_states.pt")


def get_all_optim_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, f"*optim_states.pt")


def get_all_shard_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, f"mp_rank_*_model_states.pt")


def get_model_files_by_mp_rank(checkpoint_dir, mp_rank):
    return get_checkpoint_files(checkpoint_dir, f"*mp_rank_{mp_rank:02}_model_states.pt")


def get_optim_files_by_mp_rank(checkpoint_dir, mp_rank):
    return get_checkpoint_files(checkpoint_dir, f"*mp_rank_{mp_rank:02}_optim_states.pt")

def get_shard_by_mp_rank(checkpoint_dir, mp_rank):
    shard_paths = get_checkpoint_files(checkpoint_dir, f"mp_rank_{mp_rank:02}_model_states.pt")
    assert len(shard_paths) == 1, f"Expected 1 shard, found {len(shard_paths)} shards"
    return shard_paths[0]


def create_zero3_model_state_path(dp_rank, mp_rank):
    return f"zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02}_model_states.pt"


def create_zero3_optim_state_path(dp_rank, mp_rank):
    return f"bf16_zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02}_optim_states.pt"


def aligned_size(param, num_partitions):
    padding = padding_size(param, num_partitions)
    return param.numel() + padding


def padding_size(param, num_partitions):
    remainder = param.numel() % num_partitions
    return (num_partitions - remainder) if remainder else 0


def get_optim_mp_dp_ranks(optim_state_path: str):
    """
    Given an optim state path, return the dp and mp ranks as a tuple (dp_rank, mp_rank)
    """
    optim_pat = re.compile(r"bf16_zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states.pt")
    optim_match = optim_pat.match(os.path.basename(optim_state_path))
    assert (
        optim_match is not None
    ), f"Expected optim_state_path to match {optim_pat}, found {optim_state_path}"
    optim_dp_rank, optim_mp_rank = int(optim_match.group(1)), int(optim_match.group(2))
    return optim_mp_rank, optim_dp_rank


def get_model_mp_dp_ranks(model_state_path: str):
    """
    Given a model state path, return the dp and mp ranks as a tuple (dp_rank, mp_rank)
    """
    model_pat = re.compile(r"zero_pp_rank_(\d+)_mp_rank_(\d+)_model_states.pt")
    model_match = model_pat.match(os.path.basename(model_state_path))
    assert (
        model_match is not None
    ), f"Expected model_state_path to match {model_pat}, found {model_state_path}"
    model_dp_rank, model_mp_rank = int(model_match.group(1)), int(model_match.group(2))
    return model_mp_rank, model_dp_rank


def get_sharded_mp_rank(sharded_state_path: str):
    sharded_pat = re.compile(r"mp_rank_(\d+)_model_states.pt")
    sharded_match = sharded_pat.match(os.path.basename(sharded_state_path))
    assert (
        sharded_match is not None
    ), f"Expected sharded_state_path to match {sharded_pat}, not found in {sharded_state_path}"
    sharded_mp_rank = int(sharded_match.group(1))
    return sharded_mp_rank




def normalize_model_config(model_config: dict) -> dict:
    config = {}
    for k in model_config.keys():
        config[k.replace("-", "_")] = model_config[k]
    return config


def load_model_config(model_config_path: str) -> dict:
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    return normalize_model_config(model_config)

# def pad_to_multiple(t: torch.Tensor, multiple: int = 16, dim: int = 1) -> torch.Tensor:
#     assert dim in [0, 1], f"Expected dim to be 0 or 1, found {dim}"
#     if isinstance(t, torch.Tensor):
#         assert t.dim() == 2, f"Expected tensor to have 2 dimensions, found {t.dim()} dimensions"

#         remainder = t.size(dim) % multiple
        
#         if remainder == 0:
#             return t
#         padding = (multiple - remainder)
#         if dim == -1 or dim == 1:
#             # add padding to the right
#             pad = (0, padding)
#         else:
#             # add padding to the bottom
#             pad = (0, 0, 0, padding)
#         padded_tensor =torch.nn.functional.pad(t, pad=pad, mode="constant", value=0)
#         assert padded_tensor.size(dim) % multiple == 0, f"Expected padded tensor to have size divisible by {multiple}, found {padded_tensor.size(dim)}"
#         return padded_tensor
#     elif isinstance(t, list) or isinstance(t, tuple) or isinstance(t, torch.Size):
#         remainder = t[dim] % multiple
#         if remainder == 0:
#             return t
#         padding = (multiple - remainder)
#         padded_dim = t[dim] + padding
#         padded_shape = torch.Size((t[0], padded_dim)) if dim == 1 else torch.Size((padded_dim, t[1]))
#         assert padded_shape[dim] % multiple == 0, f"Expected padded shape to have size divisible by {multiple}, found {padded_shape[dim]}"
#         return padded_shape

#     else:
#         raise ValueError(f"Expected tensor, list, tuple, or torch.Size, found {type(t)}")

def pad_to_multiple(d: int, multiple: int = 16) -> int:
    remainder = d % multiple
    if remainder == 0:
        return d, 0
    padding = (multiple - remainder)
    return d+padding, padding

def pad_to_multiple_tensor(t: torch.Tensor, multiples: Tuple[int, int] = FP8_SHAPE, dim: Tuple[int, int] = (0, 1)) -> torch.Tensor:
    assert t.dim() == 2, f"Expected tensor to have 2 dimensions, found {t.dim()} dimensions"
    assert dim in [(0, 1), (1, 0)], f"Expected dim to be 0 or 1, found {dim}"
    _, bottom_padding = pad_to_multiple(t.size(dim[0]), multiples[0])
    _, right_padding = pad_to_multiple(t.size(dim[1]), multiples[1])
    padding = (0, right_padding, 0, bottom_padding)
    return torch.nn.functional.pad(t, pad=padding, mode="constant", value=0)
    
def pad_to_multiple_shape(shape: Tuple[int, int], multiples: Tuple[int, int] = FP8_SHAPE, dim: Tuple[int, int] = (0, 1)) -> Tuple[int, int]:
    assert dim in [(0, 1), (1, 0)], f"Expected dim to be 0 or 1, found {dim}"
    _, bottom_padding = pad_to_multiple(shape[dim[0]], multiples[0])
    _, right_padding = pad_to_multiple(shape[dim[1]], multiples[1])
    return torch.Size((shape[0] + bottom_padding, shape[1] + right_padding))

def get_model_file_by_mp_dp_rank(paths, mp_rank, dp_rank):
    rank_file = [
        p
        for p in paths
        if os.path.basename(p) == f"zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02}_model_states.pt"
    ]
    assert len(rank_file) == 1, f"Expected 1 model file, found {len(rank_file)} model files"
    return rank_file[0]


def get_optim_file_by_mp_dp_rank(paths, mp_rank, dp_rank):
    rank_file = [
        p
        for p in paths
        if os.path.basename(p) == f"bf16_zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02}_optim_states.pt"
    ]
    assert len(rank_file) == 1, f"Expected 1 optim file, found {len(rank_file)} optim files"
    return rank_file[0]
    return rank_file[0]
