#!/usr/bin/env python
"""
Steps
Buffers (zero*model_state.pt files)
- Extend explicit single decay filter `mixer.filter.decay` from [num_groups // mp_size, seq_len] to [num_groups // mp_size, target_seq_len]
- Extend implicit modal filter `mixer.filter.t` from [1, 1, seq_len] to [1, 1, target_seq_len]

Params
- Extend explicit single decay filter `mixer.filter.h` from [num_groups // mp_size, seq_len] to [num_groups // mp_size, target_seq_len]
- This requires iterating through fp32 flat groups for each dp / mp rank (8 * 256 = 2048) for 40b checkpoint
    - Concatenate all params up to each filter
    - Extend the filter by 1) reshaping to sliced flat param to shape [num_groups // mp_size, seq_len] and 2) creating new param with shape [num_groups // mp_size, target_seq_len]
    - Flatten the new param and concatenate with the existing params
    - Repeat until all filters are extended
    - Concatenate with remaining params
"""

import argparse
import glob
import math
import multiprocessing
import os
import re
import time
from collections import defaultdict

import torch
import yaml
from einops import rearrange

SOURCE_SEQ_LEN = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
TARGET_SEQ_LENS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
NUM_GROUPS = 512

PARAM_PATTERNS = ["mixer.mixer.filter.h"]  # explicit single decay filter
BUFFER_PATTERNS = [
    "mixer.mixer.filter.decay",  # explicit single decay filter
    "mixer.mixer.filter.t",
]  # implicit modal filter

EXPLICIT_SINGLE_DECAY_BUFFER = "mixer.mixer.filter.decay"
IMPLICIT_MODAL_BUFFER = "mixer.mixer.filter.t"

EXPLICIT_SINGLE_DECAY_PARAM = "mixer.mixer.filter.h"

DEVICE = "cpu"
OPTIMIZER_STATE_KEY = "optimizer_state_dict"
FP32_FLAT_GROUPS_KEY = "fp32_flat_groups"
PARTITION_COUNT_KEY = "partition_count"
DS_CONFIG_KEY = "ds_config"
DS_VERSION_KEY = "ds_version"
ZERO_STAGE_KEY = "zero_stage"
PARAM_SHAPE_KEY = "param_shapes"
MODEL_KEY = "module"
MODEL_OPTIMIZER_KEY = "optimizer"
FROZEN_PARAM_SHAPE_KEY = "frozen_param_shapes"
SHARED_PARAMS_KEY = "shared_params"
MODEL_STATE_KEYS = [
    MODEL_KEY,
    MODEL_OPTIMIZER_KEY,
    PARAM_SHAPE_KEY,
    FROZEN_PARAM_SHAPE_KEY,
    SHARED_PARAMS_KEY,
]
ALL_MODEL_STATE_KEYS = [
    "module",
    "buffer_names",
    "optimizer",
    "param_shapes",
    "frozen_param_shapes",
    "shared_params",
    "frozen_param_fragments",
    "lr_scheduler",
    "data_sampler",
    "random_ltd",
    "sparse_tensor_module_names",
    "skipped_steps",
    "global_steps",
    "global_samples",
    "dp_world_size",
    "mp_world_size",
    "ds_config",
    "ds_version",
    "iteration",
    "args",
    "data_loading",
    "random_rng_state",
    "np_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
]

EXTRA_MODEL_STATE_KEYS = [
    "ds_config",
    "args",
    # "data_sampler",
    # "random_ltd",
    # "skipped_steps",
    # "global_steps",
    # "global_samples",
    "iteration",
    "data_loading",
    "random_rng_state",
    "np_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
]


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


def zero3_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def get_model_state_paths(ds_checkpoint_dir: str) -> list[str]:
    model_state_paths = get_checkpoint_files(ds_checkpoint_dir, f"zero_*_model_states.pt")
    return model_state_paths


def get_optim_state_paths(ds_checkpoint_dir: str) -> list[str]:
    optim_state_paths = get_checkpoint_files(ds_checkpoint_dir, f"bf16_*_optim_states.pt")
    return optim_state_paths


# def check_model_states(
#     output_dir: str,
#     num_groups: int,
#     mp_size: int,
#     target_seq_len: int,
#     operator_counts: dict,
#     updated_lr_scheduler: dict,
#     verbose: bool = False,
#     remove_extra_states: bool = False,
#     reset_global_state: bool = False,
# ):
#     print(f"{os.getpid()}: Checking updated buffers in {output_dir}", flush=True)

#     expected_explicit_filter_shape = torch.Size([num_groups // mp_size, target_seq_len])
#     expected_implicit_filter_shape = torch.Size([1, 1, target_seq_len])

#     expected_explicit_filter_counts = operator_counts["hyena_medium_conv"]
#     expected_implicit_filter_counts = operator_counts["hyena"]

#     for model_state_path in get_model_state_paths(output_dir):
#         if verbose:
#             print(f"{os.getpid()}: Checking {model_state_path}")
#         model_state = torch.load(model_state_path, map_location=DEVICE)
#         model_dict = model_state["module"]
#         num_explicit_filters = 0
#         num_implicit_filters = 0
#         for k, v in model_dict.items():
#             if EXPLICIT_SINGLE_DECAY_BUFFER in k:
#                 if verbose:
#                     print(f"{os.getpid()}:  -> Checking {k}")
#                 assert (
#                     v.shape == expected_explicit_filter_shape
#                 ), f"{os.getpid()}: Expected {expected_explicit_filter_shape}, found {v.shape}"
#                 num_explicit_filters += 1
#             elif IMPLICIT_MODAL_BUFFER in k:
#                 if verbose:
#                     print(f"{os.getpid()}:  -> Checking {k}")
#                 assert (
#                     v.shape == expected_implicit_filter_shape
#                 ), f"{os.getpid()}: Expected {expected_implicit_filter_shape}, found {v.shape}"
#                 num_implicit_filters += 1

#         assert (
#             num_explicit_filters == expected_explicit_filter_counts
#         ), f"{os.getpid()}: Expected {expected_explicit_filter_counts} explicit filters, found {num_explicit_filters}"
#         assert (
#             num_implicit_filters == expected_implicit_filter_counts
#         ), f"{os.getpid()}: Expected {expected_implicit_filter_counts} implicit filters, found {num_implicit_filters}"

#         # Check param shapes updated for filter.h
#         param_shapes = model_state[PARAM_SHAPE_KEY]
#         for param_shape in param_shapes:
#             for name in param_shape.keys():
#                 if EXPLICIT_SINGLE_DECAY_PARAM in name:
#                     assert (
#                         param_shape[name] == expected_explicit_filter_shape
#                     ), f"Expected {expected_explicit_filter_shape}, found {param_shape[name]}"

#         # Check extra keys removed
#         if remove_extra_states:
#             assert (
#                 set(model_state.keys()) & set(EXTRA_MODEL_STATE_KEYS) == set()
#             ), f"{os.getpid()}: Expected {EXTRA_MODEL_STATE_KEYS} to be removed, found in {model_state}"
#         if reset_global_state:
#             assert (
#                 model_state["global_steps"] == 0
#             ), f"{os.getpid()}: Expected global_steps to be 0, found {model_state['global_steps']}"
#             assert (
#                 model_state["global_samples"] == 0
#             ), f"{os.getpid()}: Expected global_samples to be 0, found {model_state['global_samples']}"
#             assert (
#                 model_state["skipped_steps"] == 0
#             ), f"{os.getpid()}: Expected skipped_steps to be 0, found {model_state['skipped_steps']}"
#             assert (
#                 model_state["data_sampler"] is None
#             ), f"{os.getpid()}: Expected data_sampler to be None, found {model_state['data_sampler']}"
#             assert (
#                 model_state["random_ltd"] is None
#             ), f"{os.getpid()}: Expected random_ltd to be None, found {model_state['random_ltd']}"
#         assert (
#             model_state["lr_scheduler"] == updated_lr_scheduler
#         ), f"{os.getpid()}: Expected lr_scheduler to be {updated_lr_scheduler}, found {model_state['lr_scheduler']}"

#     print(f"{os.getpid()}: Model state checks passed!", flush=True)


def check_model_state(
    model_state_path: str,
    num_groups: int,
    mp_size: int,
    target_seq_len: int,
    operator_counts: dict,
    updated_lr_scheduler: dict,
    verbose: bool = False,
    remove_extra_states: bool = False,
    reset_global_state: bool = False,
):
    print(f"{os.getpid()}: Checking updated buffers in {model_state_path}", flush=True)

    expected_explicit_filter_shape = torch.Size([num_groups // mp_size, target_seq_len])
    expected_implicit_filter_shape = torch.Size([1, 1, target_seq_len])

    expected_explicit_filter_counts = operator_counts["hyena_medium_conv"]
    expected_implicit_filter_counts = operator_counts["hyena"]

    model_state = torch.load(model_state_path, map_location=DEVICE)
    model_dict = model_state["module"]
    num_explicit_filters = 0
    num_implicit_filters = 0

    for k, v in model_dict.items():
        if EXPLICIT_SINGLE_DECAY_BUFFER in k:
            if verbose:
                print(f"  -> Checking {k}")
            assert (
                v.shape == expected_explicit_filter_shape
            ), f"{os.getpid()}: Expected {expected_explicit_filter_shape}, found {v.shape}"
            num_explicit_filters += 1
        elif IMPLICIT_MODAL_BUFFER in k:
            if verbose:
                print(f"  -> Checking {k}")
            assert (
                v.shape == expected_implicit_filter_shape
            ), f"{os.getpid()}: Expected {expected_implicit_filter_shape}, found {v.shape}"
            num_implicit_filters += 1

    assert (
        num_explicit_filters == expected_explicit_filter_counts
    ), f"{os.getpid()}: Expected {expected_explicit_filter_counts} explicit filters, found {num_explicit_filters}"
    assert (
        num_implicit_filters == expected_implicit_filter_counts
    ), f"{os.getpid()}: Expected {expected_implicit_filter_counts} implicit filters, found {num_implicit_filters}"

    # Check param shapes updated for filter.h
    param_shapes = model_state[PARAM_SHAPE_KEY]
    for param_shape in param_shapes:
        for name in param_shape.keys():
            if EXPLICIT_SINGLE_DECAY_PARAM in name:
                assert (
                    param_shape[name] == expected_explicit_filter_shape
                ), f"{os.getpid()}: Expected {expected_explicit_filter_shape}, found {param_shape[name]}"

    # Check extra keys removed
    if remove_extra_states:
        assert (
            set(model_state.keys()) & set(EXTRA_MODEL_STATE_KEYS) == set()
        ), f"{os.getpid()}: Expected {EXTRA_MODEL_STATE_KEYS} to be removed, found in {model_state}"

    if reset_global_state:
        assert (
            model_state["global_steps"] == 0
        ), f"{os.getpid()}: Expected global_steps to be 0, found {model_state['global_steps']}"
        assert (
            model_state["global_samples"] == 0
        ), f"{os.getpid()}: Expected global_samples to be 0, found {model_state['global_samples']}"
        assert (
            model_state["skipped_steps"] == 0
        ), f"{os.getpid()}: Expected skipped_steps to be 0, found {model_state['skipped_steps']}"
        assert (
            model_state["data_sampler"] is None
        ), f"{os.getpid()}: Expected data_sampler to be None, found {model_state['data_sampler']}"
        assert (
            model_state["random_ltd"] is None
        ), f"{os.getpid()}: Expected random_ltd to be None, found {model_state['random_ltd']}"
    assert (
        model_state["lr_scheduler"] == updated_lr_scheduler
    ), f"{os.getpid()}: Expected lr_scheduler to be {updated_lr_scheduler}, found {model_state['lr_scheduler']}"

    print(f"{os.getpid()}: {os.path.basename(model_state_path)}: Model state checks passed!", flush=True)


def update_single_model_state(
    model_state_path: str,
    source_seq_len: int,
    target_seq_len: int,
    output_dir: str,
    remove_extra_states: bool,
    reset_global_state: bool,
    verbose: bool,
    target_model_config: dict,  # Pass target_model_config as argument
    expected_explicit_filter_shape: torch.Size,
    expected_implicit_filter_shape: torch.Size,
    expected_explicit_filter_counts: int,
    expected_implicit_filter_counts: int,
    target_explicit_filter_shape: torch.Size,
    target_implicit_filter_shape: torch.Size,
    num_groups: int,
    mp_size: int,
    operator_counts: dict,
    start_mp_rank: int,
    end_mp_rank: int,
):
    if verbose:
        print(f"{os.getpid()}: processing {model_state_path}", flush=True)

    start_time = time.time()
    model_mp_rank, model_dp_rank = get_model_mp_dp_ranks(model_state_path)

    # if model_mp_rank < start_mp_rank or model_mp_rank > end_mp_rank:
    #     print(f"{os.getpid()}: Skipping {os.path.basename(model_state_path)}: {model_mp_rank} {model_dp_rank}")
    #     return
    
    print(f"{os.getpid()}: Updating model state {os.path.basename(model_state_path)}: {model_mp_rank} {model_dp_rank}", flush=True)
    # Load and potentially modify model state (same as before)
    model_state = torch.load(model_state_path, map_location=DEVICE)

    if remove_extra_states:
        keys = list(model_state.keys())
        for k in keys:
            if k in EXTRA_MODEL_STATE_KEYS:
                model_state.pop(k)

    if reset_global_state:
        model_state["global_steps"] = 0
        model_state["global_samples"] = 0
        model_state["skipped_steps"] = 0
        model_state["data_sampler"] = None
        model_state["random_ltd"] = None

    model_dict = model_state["module"]
    explicit_filter_counts = 0
    implicit_filter_counts = 0

    # Update buffers
    for k, v in model_dict.items():
        if EXPLICIT_SINGLE_DECAY_BUFFER in k:
            if verbose:
                print(
                    f"{os.getpid()}:  {k}: reshaping from {expected_explicit_filter_shape} to {target_explicit_filter_shape}",
                    flush=True,
                )
            assert (
                v.shape == expected_explicit_filter_shape
            ), f"{os.getpid()}: Expected {expected_explicit_filter_shape}, found {v.shape}"

            new_w = torch.zeros(target_explicit_filter_shape, dtype=v.dtype, device=v.device)
            new_w[:, :source_seq_len] = v
            assert (
                new_w.shape == target_explicit_filter_shape
            ), f"{os.getpid()}: Expected {target_explicit_filter_shape}, found {new_w.shape}"
            assert new_w[:, :source_seq_len].equal(
                v
            ), f"{os.getpid()}: Expected {v}, found {new_w[:, :source_seq_len]}"
            model_dict[k] = new_w

            explicit_filter_counts += 1

        elif IMPLICIT_MODAL_BUFFER in k:
            if verbose:
                print(
                    f"{os.getpid()}:  {k}: reshaping from {expected_implicit_filter_shape} to {target_implicit_filter_shape}",
                    flush=True,
                )
            assert (
                v.shape == expected_implicit_filter_shape
            ), f"{os.getpid()}: Expected {expected_implicit_filter_shape}, found {v.shape}"
            new_w = rearrange(
                torch.arange(target_seq_len, dtype=torch.float32, device=v.device), "L -> 1 1 L"
            )
            assert (
                new_w.shape == target_implicit_filter_shape
            ), f"{os.getpid()}: Expected {target_implicit_filter_shape}, found {new_w.shape}"

            model_dict[k] = new_w
            implicit_filter_counts += 1

    assert (
        explicit_filter_counts == expected_explicit_filter_counts
    ), f"{os.getpid()}: Expected {expected_explicit_filter_counts} explicit filters, found {explicit_filter_counts}"

    assert (
        implicit_filter_counts == expected_implicit_filter_counts
    ), f"{os.getpid()}: Expected {expected_implicit_filter_counts} implicit filters, found {implicit_filter_counts}"

    # Update param shapes
    param_shapes = model_state[PARAM_SHAPE_KEY]
    for param_shape in param_shapes:
        for name in param_shape.keys():
            if EXPLICIT_SINGLE_DECAY_PARAM in name:
                print(
                    f"{os.getpid()}:  -> Updating param shape {name} from {param_shape[name]} to {target_explicit_filter_shape}",
                    flush=True,
                )
                param_shape[name] = target_explicit_filter_shape
    model_state[PARAM_SHAPE_KEY] = param_shapes

    # Update lr_scheduler
    updated_lr_scheduler = {}
    updated_lr_scheduler["min_lr"] = target_model_config["min_lr"]
    updated_lr_scheduler["start_lr"] = target_model_config["optimizer"]["params"]["lr"]
    updated_lr_scheduler["num_iters"] = 0
    updated_lr_scheduler["end_iter"] = target_model_config["train_iters"]
    updated_lr_scheduler["decay_style"] = target_model_config["lr_decay_style"]
    updated_lr_scheduler["warmup_iter"] = int(
        target_model_config["warmup"] * target_model_config["train_iters"]
    )

    assert (
        model_state["lr_scheduler"].keys() == updated_lr_scheduler.keys()
    ), f"{os.getpid()}: Expected lr_scheduler to have keys {updated_lr_scheduler.keys()}, found {model_state['lr_scheduler'].keys()}"
    model_state["lr_scheduler"] = updated_lr_scheduler

    end_time = time.time()
    duration = end_time - start_time
    print(f"{os.getpid()}: Finished updating model state {os.path.basename(model_state_path)} in {duration:.2f} seconds", flush=True)

    # Save updated model state
    start_time = time.time()
    output_path = os.path.join(output_dir, os.path.basename(model_state_path))
    torch.save(model_state, output_path)
    end_time = time.time()
    duration = end_time - start_time
    print(f"{os.getpid()}: Finished saving updated model state {os.path.basename(model_state_path)} in {duration:.2f} seconds", flush=True)

    if verbose:
        print(f"{os.getpid()}: saved updated model state to {output_path}")

    start_time = time.time()
    check_model_state(
        model_state_path=output_path,
        num_groups=num_groups,
        mp_size=mp_size,
        target_seq_len=target_seq_len,
        operator_counts=operator_counts,
        updated_lr_scheduler=updated_lr_scheduler,
        verbose=verbose,
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f"{os.getpid()}: Finished checking updated model state {os.path.basename(output_path)} in {duration:.2f} seconds", flush=True)


def update_model_states(
    num_groups: int,
    mp_size: int,
    source_seq_len: int,
    target_seq_len: int,
    model_state_paths: list[str],
    source_model_config: dict,
    target_model_config: dict,
    output_dir: str,
    start_mp_rank: int,
    end_mp_rank: int,
    remove_extra_states: bool,
    reset_global_state: bool,
    verbose: bool = False,  #
    num_workers: int = 1,
):
    # Check operator counts match
    source_operator_counts = parse_operator_config(source_model_config)
    target_operator_counts = parse_operator_config(target_model_config)
    assert (
        sum(source_operator_counts.values()) == sum(target_operator_counts.values())
    ), f"Expected source and target operator counts to match, found {source_operator_counts} and {target_operator_counts}"

    operator_counts = target_operator_counts
    # Operator count checks
    expected_explicit_filter_counts = operator_counts["hyena_medium_conv"]
    expected_implicit_filter_counts = operator_counts["hyena"]

    # Shape checks
    expected_explicit_filter_shape = torch.Size([num_groups // mp_size, source_seq_len])
    expected_implicit_filter_shape = torch.Size([1, 1, source_seq_len])
    target_explicit_filter_shape = torch.Size([num_groups // mp_size, target_seq_len])
    target_implicit_filter_shape = torch.Size([1, 1, target_seq_len])
    print(
        f"Updating {EXPLICIT_SINGLE_DECAY_BUFFER} from {expected_explicit_filter_shape} to {target_explicit_filter_shape}, {IMPLICIT_MODAL_BUFFER} from {expected_implicit_filter_shape} to {target_implicit_filter_shape}",
        flush=True,
    )

    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            args = [
                (
                    path,
                    source_seq_len,
                    target_seq_len,
                    output_dir,
                    remove_extra_states,
                    reset_global_state,
                    verbose,
                    target_model_config,
                    expected_explicit_filter_shape,
                    expected_implicit_filter_shape,
                    expected_explicit_filter_counts,
                    expected_implicit_filter_counts,
                    target_explicit_filter_shape,
                    target_implicit_filter_shape,
                    num_groups,
                    mp_size,
                    operator_counts,
                    start_mp_rank,
                    end_mp_rank,
                )
                for path in model_state_paths
            ]
            pool.starmap(update_single_model_state, args)
    else:
        for model_state_path in model_state_paths:
            update_single_model_state(
                model_state_path=model_state_path,
                source_seq_len=source_seq_len,
                target_seq_len=target_seq_len,
                output_dir=output_dir,
                remove_extra_states=remove_extra_states,
                reset_global_state=reset_global_state,
                verbose=verbose,
                target_model_config=target_model_config,
                expected_explicit_filter_shape=expected_explicit_filter_shape,
                expected_implicit_filter_shape=expected_implicit_filter_shape,
                expected_explicit_filter_counts=expected_explicit_filter_counts,
                expected_implicit_filter_counts=expected_implicit_filter_counts,
                target_explicit_filter_shape=target_explicit_filter_shape,
                target_implicit_filter_shape=target_implicit_filter_shape,
                num_groups=num_groups,
                mp_size=mp_size,
                operator_counts=operator_counts,
                start_mp_rank=start_mp_rank,
                end_mp_rank=end_mp_rank,
            )

    print(f"Saved {len(model_state_paths)} updated model states to {output_dir}")

def get_optim_mp_dp_ranks(optim_state_path: str):
    optim_pat = re.compile(r"bf16_zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states.pt")
    optim_match = optim_pat.match(os.path.basename(optim_state_path))
    assert (
        optim_match is not None
    ), f"Expected optim_state_path to match {optim_pat}, found {optim_state_path}"
    optim_dp_rank, optim_mp_rank = int(optim_match.group(1)), int(optim_match.group(2))
    return optim_mp_rank, optim_dp_rank


def get_model_mp_dp_ranks(model_state_path: str):
    model_pat = re.compile(r"zero_pp_rank_(\d+)_mp_rank_(\d+)_model_states.pt")
    model_match = model_pat.match(os.path.basename(model_state_path))
    assert (
        model_match is not None
    ), f"Expected model_state_path to match {model_pat}, found {model_state_path}"
    model_dp_rank, model_mp_rank = int(model_match.group(1)), int(model_match.group(2))
    return model_mp_rank, model_dp_rank

def check_model_optim_path(model_state_path: str, optim_state_path: str, verbose: bool = False):
    # Extract dp_rank and mp_rank from optim_state_path
    optim_mp_rank, optim_dp_rank = get_optim_mp_dp_ranks(optim_state_path)
    
    # Extract dp_rank and mp_rank from model_state_path
    model_mp_rank, model_dp_rank = get_model_mp_dp_ranks(model_state_path)

    assert (
        optim_dp_rank == model_dp_rank
    ), f"Expected dp_rank to be the same, found {optim_dp_rank} and {model_dp_rank}"
    assert (
        optim_mp_rank == model_mp_rank
    ), f"Expected mp_rank to be the same, found {optim_mp_rank} and {model_mp_rank}"

    if verbose:
        print(f"Checking {model_state_path} and {optim_state_path}")
        print(f"  -> dp_rank: {optim_dp_rank} == {model_dp_rank}")
        print(f"  -> mp_rank: {optim_mp_rank} == {model_mp_rank}")


def check_optimizer_states(
    optim_state_path: str,
    output_path: str,
    source_partitioned_numel: int,
    target_partitioned_numel: int,
    source_seq_len: int,
    target_seq_len: int,
    added_numels: list[int],
    new_numels: list[int],
    filter_offsets_per_state: list[list[int]],
    model_config: dict,
    reset_optim_states: bool = False,
    verbose: bool = False,
):
    optim_state = torch.load(optim_state_path, map_location=DEVICE)
    output_state = torch.load(output_path, map_location=DEVICE)
    prev_fp32_flat_groups = optim_state[OPTIMIZER_STATE_KEY][FP32_FLAT_GROUPS_KEY]
    new_fp32_flat_groups = output_state[OPTIMIZER_STATE_KEY][FP32_FLAT_GROUPS_KEY]

    assert len(prev_fp32_flat_groups) == len(
        new_fp32_flat_groups
    ), f"{os.getpid()}: Expected {len(prev_fp32_flat_groups)} param groups, found {len(new_fp32_flat_groups)}"
    assert len(added_numels) == len(
        new_numels
    ), f"{os.getpid()}: Expected {len(added_numels)} added numels, found {len(new_numels)}"
    assert len(filter_offsets_per_state) == len(
        prev_fp32_flat_groups
    ), f"{os.getpid()}: Expected {len(prev_fp32_flat_groups)} filter offsets, found {len(filter_offsets_per_state)}"

    # Check overall numels
    for i, (added_numel, new_numel) in enumerate(zip(added_numels, new_numels)):
        if verbose:
            print(
                f"{os.getpid()}:  -> param group {i}: {prev_fp32_flat_groups[i].shape} -> {new_fp32_flat_groups[i].shape}", flush=True
            )
        assert (
            new_fp32_flat_groups[i].numel() == new_numel
        ), f"{os.getpid()}: Expected {new_numel} numels, found {new_fp32_flat_groups[i].numel()}"
        assert (
            new_fp32_flat_groups[i].numel() == prev_fp32_flat_groups[i].numel() + added_numel
        ), f"{os.getpid()}: Expected {prev_fp32_flat_groups[i].numel() + added_numel} numels, found {new_fp32_flat_groups[i].numel()}"

    # Fine-grained check: check param equivalence outside of filter.h segments
    # Check equivalence up to source_seq_len in filter.h segments and zeros up to target_seq_len
    for i in range(len(filter_offsets_per_state)):

        prev_fp32_flat_group = prev_fp32_flat_groups[i]
        new_fp32_flat_group = new_fp32_flat_groups[i]
        if len(filter_offsets_per_state[i]) == 0:
            assert prev_fp32_flat_group.equal(
                new_fp32_flat_group
            ), f"{os.getpid()}: Expected param groups {i} to be equal since no filter.h params in this param group"
            continue

        # Segment flat params from offsets of filter.h params
        # E.g., if 1 filter.h offset at 10 and fp32_flat_group.numel() = 100, then offsets should be [0, 10, 10 + expected_partitioned_shape.numel(), 100]
        # new_offsets should be [0, 10, 10 + target_partitioned_shape.numel(), 100 + target_partitioned_shape.numel()]
        # if 2 filter.h params, the [0, offset[0], offset[0] + expected_partitioned_shape.numel(), offset[1], offset[1] + expected_partitioned_shape.numel(), fp32_flat_group.numel()]
        prev_segments = [0]
        for filter_offset in filter_offsets_per_state[i]:
            prev_segments.append(filter_offset)
            prev_segments.append(filter_offset + source_partitioned_numel)  #expected_partitioned_shape.numel())
        if prev_segments[-1] != prev_fp32_flat_group.numel():
            prev_segments.append(prev_fp32_flat_group.numel())

        new_segments = [0]
        extended_offset = 0
        for filter_offset in filter_offsets_per_state[i]:
            new_segments.append(filter_offset + extended_offset)
            new_segments.append(filter_offset + extended_offset + target_partitioned_numel)
            extended_offset += target_partitioned_numel - source_partitioned_numel
        if new_segments[-1] != new_fp32_flat_group.numel():
            new_segments.append(new_fp32_flat_group.numel())

        # Check equivalence outside filter.h regions
        for j in range(len(prev_segments) - 1):
            prev_segment = prev_fp32_flat_group.narrow(
                0, prev_segments[j], prev_segments[j + 1] - prev_segments[j]
            )
            new_segment = new_fp32_flat_group.narrow(
                0, new_segments[j], new_segments[j + 1] - new_segments[j]
            )
            if prev_segments[j] in filter_offsets_per_state[i]:
                # Check previous filter.h params were configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3.ymlcorrectly copied
                _, dp_rank = get_optim_mp_dp_ranks(optim_state_path)
                
                # if dp_rank % 4 == 0:
                assert new_segment[:source_partitioned_numel].equal(prev_segment)
                # else:
                #     assert new_segment.equal(torch.zeros_like(new_segment))

                # old_filter = prev_segment.reshape(expected_partitioned_shape)
                # new_filter = new_segment.reshape(target_partitioned_shape)
                # assert new_filter[:, :source_seq_len].equal(
                #     old_filter), f"{os.getpid()}: Expected filter.h region to be equal: {old_filter}, found {new_filter[:,:source_seq_len]}"
                # assert new_filter[:, source_seq_len:].equal(
                #     torch.zeros_like(new_filter[:, source_seq_len:])
                # ), f"{os.getpid()}: Expected filter.h beyond {source_seq_len} to be zero: {new_filter[:,source_seq_len:]}"
            else:
                assert prev_segment.equal(
                    new_segment
                ), f"Expected param groups to be equal outside filter.h regions: {prev_segments[j]}:{prev_segments[j+1]} != {new_segments[j]}:{new_segments[j+1]}"

    # Check updated optimizer states
    if reset_optim_states:
        optimizer_state = output_state[OPTIMIZER_STATE_KEY][OPTIMIZER_STATE_KEY]
        for param_group in optimizer_state["param_groups"]:
            assert param_group["step"] == 0, f"Expected step to be 0, found {param_group['step']}"
            assert (
                param_group["lr"] == target_model_config["optimizer"]["params"]["lr"]
            ), f"{os.getpid()}: Expected lr to be {target_model_config['optimizer']['params']['lr']}, found {param_group['lr']}"
            assert (
                param_group["betas"] == target_model_config["optimizer"]["params"]["betas"]
            ), f"{os.getpid()}: Expected betas to be {target_model_config['optimizer']['params']['betas']}, found {param_group['betas']}"
            assert (
                param_group["eps"] == target_model_config["optimizer"]["params"]["eps"]
            ), f"{os.getpid()}: Expected eps to be {target_model_config['optimizer']['params']['eps']}, found {param_group['eps']}"
        optimizer = optimizer_state["state"]

        optimizer_params = optimizer.values()   
        for i, (numel, params) in enumerate(zip(new_numels, optimizer_params)):
            assert params["exp_avg"].numel() == numel, f"{os.getpid()}: Expected {numel} exp_avg numels, found {params['exp_avg'].numel()}"
            assert params["exp_avg_sq"].numel() == numel, f"{os.getpid()}: Expected {numel} exp_avg_sq numels, found {params['exp_avg_sq'].numel()}"
            assert params["exp_avg"].equal(torch.zeros_like(params["exp_avg"])), f"{os.getpid()}: Expected exp_avg to be zero, found {params['exp_avg']}"
            assert params["exp_avg_sq"].equal(torch.zeros_like(params["exp_avg_sq"])), f"{os.getpid()}: Expected exp_avg_sq to be zero, found {params['exp_avg_sq']}"

def update_optimizer_states_single(
    model_state_path: str,
    optim_state_path: str,
    target_model_config: dict,
    output_dir: str,
    expected_explicit_filter_counts: int,
    source_seq_len: int,
    target_seq_len: int,
    num_groups: int,
    mp_size: int,
    dp_size: int,
    reset_optim_states: bool,
    remove_extra_optim_states: bool,
    start_mp_rank: int,
    end_mp_rank: int,
    verbose: bool = False,
):
    if verbose:
        print(f"{os.getpid()}: Processing {model_state_path} and {optim_state_path}", flush=True)
    start_time = time.time()
    check_model_optim_path(model_state_path, optim_state_path, verbose=verbose)

    mp_rank, dp_rank = get_optim_mp_dp_ranks(optim_state_path)
    
    # if mp_rank < start_mp_rank or mp_rank > end_mp_rank:
    #     print(f"{os.getpid()}: Skipping optim state {os.path.basename(optim_state_path)}: mp_rank: {mp_rank}, dp_rank: {dp_rank}")
    #     return
    
    print(f"{os.getpid()}: Updating optim state {os.path.basename(optim_state_path)} mp_rank: {mp_rank}, dp_rank: {dp_rank}", flush=True)
    optim_state = torch.load(optim_state_path, map_location=DEVICE)

    # Remove ds_config since this should be loaded from model config
    if DS_CONFIG_KEY in optim_state:
        optim_state.pop(DS_CONFIG_KEY)
    optimizer_state_dict = optim_state[OPTIMIZER_STATE_KEY]

    partition_counts = optimizer_state_dict[PARTITION_COUNT_KEY]
    assert partition_counts == dp_size, f"{os.getpid()}: Expected {dp_size} model partitions, found {partition_counts}"

    model_state = torch.load(model_state_path, map_location=DEVICE)
    param_shapes = model_state[PARAM_SHAPE_KEY]
    fp32_flat_groups = optimizer_state_dict[FP32_FLAT_GROUPS_KEY]

    assert len(param_shapes) == len(
        fp32_flat_groups
    ), f"{os.getpid()}: Expected {len(param_shapes)} param groups, found {len(fp32_flat_groups)}"

    # Per optim state path vars
    num_filter_counts = 0
    filter_offsets_per_state = []
    new_numels = []
    added_numels = []

    # Partitioned filter.h numels
    source_partitioned_numel = (num_groups // mp_size) * source_seq_len // dp_size
    target_partitioned_numel = (num_groups // mp_size) * target_seq_len // dp_size

    for i, (param_shape, fp32_flat_group) in enumerate(zip(param_shapes, fp32_flat_groups)):
        # Per param group vars
        filter_offsets_per_group = []
        offset = 0
        total_numel = 0
        total_params = 0
        wanted_numel = 0
        for name, p_shape in param_shape.items():
            wanted_numel += p_shape.numel()
        available_numel = fp32_flat_group.numel()

        print(f"{os.getpid()}:  -> param group{i}: wanted {wanted_numel} numels, available {available_numel * dp_size} numels", flush=True)
        assert (
            wanted_numel <= available_numel * dp_size
        ), f"{os.getpid()}: Expected {wanted_numel} numels, available {available_numel} numels"

        new_fp32_flat_group = torch.tensor([], dtype=fp32_flat_group.dtype, device=fp32_flat_group.device)
        for name, p_shape in param_shape.items():
            unpartitioned_numel = p_shape.numel()
            total_numel += unpartitioned_numel
            total_params += 1
            partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(
                unpartitioned_numel, dp_size
            )
            if partitioned_padding_numel > 0:
                print(
                    f"{os.getpid()}:  -> {name}: {unpartitioned_numel=} = {partitioned_numel=}  {partitioned_padding_numel=} {dp_size=}"
                )
            # assert (
            #     partitioned_padding_numel == 0
            # ), f"{os.getpid()}: {name}: Expected no padding, found {partitioned_padding_numel}"
            partitioned_param = fp32_flat_group.narrow(0, offset, partitioned_numel)
            assert (
                partitioned_param.numel() == (p_shape.numel() + partitioned_padding_numel) // dp_size
            ), f"{os.getpid()}: {name}: Expected {(p_shape.numel() + partitioned_padding_numel) // dp_size} numels, found {partitioned_param.numel()}"

            if EXPLICIT_SINGLE_DECAY_PARAM in name:
                print(f"{os.getpid()}:  -> Extending {name} from source_partitioned_numel {source_partitioned_numel} to target_partitioned_numel {target_partitioned_numel}", flush=True
                )
                # for seq_len 8192, num_groups 512
                # each filter will be shape [64, 8192]
                # For mp_size = 8, dp_size = 256, this means each dp rank will have 2048 elements
                # Therefore, 4 ranks span one row since 8192 / 2048 = 4.  When extending, [64, 8192 * factor] => 64 * 8192 * factor // 256 = 2048 * factor, (8192 * factor) // (2048 * factor) = 4, so 4 ranks still span one row
                # so only dp_rank % 4 should copy filter.h params
                # source_partitioned_numel = (num_groups // mp_size) * source_seq_len // dp_size -> implies that when extending, each rank will store target_seq_len // source_seq_len more elements but span the same number of rows
                # Make the simplifying assumption that the entire conv filter will fit in the partitioned numel of a single dp rank
                assert source_partitioned_numel >= target_model_config["hyena_medium_conv_len"], f"{os.getpid()}: Partitioned numels {source_partitioned_numel} < {target_model_config['hyena_medium_conv_len']}"
                assert source_partitioned_numel <= source_seq_len, f"{os.getpid()}: Partitioned numels {source_partitioned_numel} > source_seq_len {source_seq_len}, reconstruction only works if partitioned numels is less than source_seq_len"
                assert source_seq_len % source_partitioned_numel == 0, f"{os.getpid()}: source_seq_len {source_seq_len} must be divisible by source_partitioned_numel {source_partitioned_numel}"
                assert source_partitioned_numel == partitioned_param.numel(), f"{os.getpid()}: source_partitioned_numel {source_partitioned_numel} != partitioned_param.numel() {partitioned_param.numel()}"
                # Given above assumptions, we can simply copy over the source_partitioned_numel, since we are guaranteed that 
                # a) the source and target dp ranks will map to the same positions within the reconstructed filter
                # b) the entire filter will fit into a single partition
                # This means that even though we are copying extraneous elements, it doesn't matter since we are slicing off `hyena_medium_conv_len` elements along the sequence dimension
                
                # assert (
                #     p_shape == expected_unpartitioned_shape
                # ), f"{os.getpid()}: {name}: Expected {expected_unpartitioned_shape}, found {p_shape.shape}"
                # assert (
                #     partitioned_param.numel() == expected_partitioned_numel
                # ), f"{os.getpid()}: {name}: Expected {expected_partitioned_numel} numels, found {partitioned_param.numel()}"


                # print(f"{os.path.basename(optim_state_path)}: {dp_rank}")
                new_param = torch.zeros(target_partitioned_numel, dtype=partitioned_param.dtype, device=partitioned_param.device)
                
               # if dp_rank % 4 == 0:
                    # Technically only need to copy up to hyena_medium_conv_len, but this is a simplifying assumption
                new_param[:source_partitioned_numel] = partitioned_param.detach().clone()
          
                # per_partition_filter = partitioned_param.view(expected_partitioned_shape)
                # new_param = torch.zeros(
                #     target_partitioned_shape,
                #     dtype=partitioned_param.dtype,>
                #     device=partitioned_param.device,
                # )
                # new_param[:, :source_seq_len] = per_partition_filter
                # assert (
                #     new_param.shape == target_partitioned_shape
                # ), f"{os.getpid()}: {name}: Expected {target_partitioned_shape}, found {new_param.shape}"
                # assert new_param[:, :source_seq_len].equal(
                #     per_partition_filter
                # ), f"{os.getpid()}: {name}: Expected {per_partition_filter}, found {new_param[:, :source_seq_len]}"
                
                new_fp32_flat_group = torch.cat((new_fp32_flat_group, new_param.view(-1)), 0)
                num_filter_counts += 1
                filter_offsets_per_group.append(offset)
            else:
                new_fp32_flat_group = torch.cat((new_fp32_flat_group, partitioned_param), 0)
            if verbose:
                print(f"  -> {name}: {new_fp32_flat_group.shape}")
            offset += partitioned_numel

        new_numel = new_fp32_flat_group.numel()
        new_numels.append(new_numel)

        added_numel = num_filter_counts * (target_partitioned_numel - source_partitioned_numel)
        added_numels.append(added_numel)
        filter_offsets_per_state.append(filter_offsets_per_group)

        assert (
            new_numel == available_numel + added_numel
        ), f"{os.getpid()}: Expected {available_numel + added_numel} numels, found {new_numel}"
        assert offset == available_numel, f"{os.getpid()}: Expected {available_numel} numels, found {offset}"
        
        if verbose:
            print(f"{os.getpid()}:  -> param group {i}: {new_fp32_flat_group.shape}", flush=True)
        optimizer_state_dict[FP32_FLAT_GROUPS_KEY][i] = new_fp32_flat_group

    assert (
        num_filter_counts == expected_explicit_filter_counts
    ), f"{os.getpid()}: Expected {expected_explicit_filter_counts} explicit filters, found {num_filter_counts}"

    # Structure of optimizer states is a dict with keys 'state' and 'param_groups'
    # 'param_groups' containers optimizer config and importantly 'step' which needs to be reset to 0
    # 'state' is a dict of dicts, keyed by param_group number, with each nested dict a dict of states 'exp_avg' and 'exp_avg_sq'
    # the states are flattened partitioned by dp_rank and need to be set to zero when finetuning
    if reset_optim_states:
        # optimizer_state_dict.keys = zero_stage', 'loss_scaler', 'dynamic_loss_scale', 'overflow', 'partition_count', 'optimizer_state_dict', 'fp32_flat_groups'
        # leave loss_scaler as is, as it's only used for fp16 training; same for dynamic_loss_scale, which should be False
        optimizer_state_dict["overflow"] = False
        optimizer_state_dict["dynamic_loss_scale"] = False
        optimizer_states = optimizer_state_dict[OPTIMIZER_STATE_KEY]

        # Reset optimizer metadata, reset lr; update betas, eps in case these changed in target_model_config
        for param_group in optimizer_states["param_groups"]:
            param_group["step"] = 0
            param_group["lr"] = target_model_config["optimizer"]["params"]["lr"]
            param_group["betas"] = target_model_config["optimizer"]["params"]["betas"]
            param_group["eps"] = target_model_config["optimizer"]["params"]["eps"]
        
        # Reset optimizer states, assumes Adam optimizer
        optimizer = optimizer_states["state"]
        assert len(optimizer) == len(new_numels), f"{os.getpid()}: Expected {len(new_numels)} optimizer param groups, found {len(optimizer)}"
        
        optimizer_params = optimizer.values()
        for i, (numel, params) in enumerate(zip(new_numels, optimizer_params)):
            params["exp_avg"] = torch.zeros(numel, dtype=params["exp_avg"].dtype, device=params["exp_avg"].device)
            params["exp_avg_sq"] = torch.zeros(numel, dtype=params["exp_avg_sq"].dtype, device=params["exp_avg_sq"].device)

        optimizer_state_dict[OPTIMIZER_STATE_KEY] = optimizer_states

    if remove_extra_optim_states:
        # Remove extraneous states
        keys = list(optimizer_state_dict.keys())
        for k in keys:
            if k not in [ZERO_STAGE_KEY, PARTITION_COUNT_KEY, FP32_FLAT_GROUPS_KEY]:
                optimizer_state_dict.pop(k)

    optim_state[OPTIMIZER_STATE_KEY] = optimizer_state_dict
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"{os.getpid()}: Finished updating optim state {os.path.basename(optim_state_path)} in {duration:.2f} seconds", flush=True)

    # Save updated optim state
    start_time = time.time()
    output_path = os.path.join(output_dir, os.path.basename(optim_state_path))
    print(f"{os.getpid()}: Saving updated optim state to {output_path}", flush=True)
    torch.save(optim_state, output_path)
    end_time = time.time()
    duration = end_time - start_time
    print(f"{os.getpid()}: Finished saving updated optim state {os.path.basename(optim_state_path)} in {duration:.2f} seconds", flush=True)

    print(f"{os.getpid()}: Checking params {optim_state_path} and {output_path}", flush=True)
    start_time = time.time()
    check_optimizer_states(
        optim_state_path,
        output_path,
        source_partitioned_numel,
        target_partitioned_numel,
        source_seq_len,
        target_seq_len,
        added_numels,
        new_numels,
        filter_offsets_per_state,
        model_config=target_model_config,
        reset_optim_states=reset_optim_states,
        verbose=verbose,
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f"{os.getpid()}: Finished checking optim state {os.path.basename(optim_state_path)} in {duration:.2f} seconds", flush=True)


def update_optimizer_states(
    num_groups: int,
    mp_size: int,
    dp_size: int,
    source_seq_len: int,
    target_seq_len: int,
    model_state_paths: list[str],
    optim_state_paths: list[str],
    source_model_config: dict,
    target_model_config: dict,
    output_dir: str,
    start_mp_rank: int,
    end_mp_rank: int ,
    remove_extra_optim_states: bool,
    reset_optim_states: bool,
    verbose: bool = False,
    num_workers: int = 1,

):
    operator_counts = parse_operator_config(target_model_config)
    expected_explicit_filter_counts = operator_counts["hyena_medium_conv"]
    groups_per_dp_rank = (num_groups // mp_size) #// dp_size

    # Each filter is first sharded along dim 0 by mp_rank then flattened and partitioned by dp_rank
    # So an original filter of shape [num_groups, seq_len]
    #  - first sharded to [num_groups // mp_size, seq_len]
    #  - then flattened to [num_groups // mp_size * seq_len]
    #  - narrowed to [rank * partition_size, (rank+1) * partition_size]
    #  - where partition_size = (num_groups // mp_size * seq_len) // dp_size

    # expected_unpartitioned_shape = torch.Size([num_groups // mp_size, source_seq_len])
    # expected_partitioned_shape = torch.Size([groups_per_dp_rank, source_seq_len])
    # target_partitioned_shape = torch.Size([groups_per_dp_rank, target_seq_len])

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            args = [
                (
                    model_state_path,
                    optim_state_path,
                    target_model_config,
                    output_dir,
                    expected_explicit_filter_counts,
                    source_seq_len,
                    target_seq_len,
                    num_groups,
                    mp_size,
                    dp_size,
                    reset_optim_states,
                    remove_extra_optim_states,
                    verbose,
                    start_mp_rank,
                    end_mp_rank,
                )
                for model_state_path, optim_state_path in zip(model_state_paths, optim_state_paths)
            ]
            pool.starmap(update_optimizer_states_single, args)
    else:
        for model_state_path, optim_state_path in zip(model_state_paths, optim_state_paths):
            if verbose:
                print(f"Processing {model_state_path} and {optim_state_path}", flush=True)

            update_optimizer_states_single(
                model_state_path=model_state_path,
                optim_state_path=optim_state_path,
                target_model_config=target_model_config,
                output_dir=output_dir,
                expected_explicit_filter_counts=expected_explicit_filter_counts,
                source_seq_len=source_seq_len,
                target_seq_len=target_seq_len,
                num_groups=num_groups,
                mp_size=mp_size,
                dp_size=dp_size,
                reset_optim_states=reset_optim_states,
                remove_extra_optim_states=remove_extra_optim_states,
                verbose=verbose,
                start_mp_rank=start_mp_rank,
                end_mp_rank=end_mp_rank,
            )


def extend_filters(
    args,
    ds_checkpoint_dir: str,
    source_model_config: dict,
    target_model_config: dict,
    output_dir: str,
    verbose: bool = False,
):
    dp_size = args.dp_size
    mp_size = args.mp_size
    world_size = args.world_size
    start_mp_rank = args.start_mp_rank
    end_mp_rank = args.end_mp_rank
    num_groups = args.num_groups
    source_seq_len = args.source_seq_len
    target_seq_len = args.target_seq_len
    remove_extra_model_states = args.remove_extra_model_states
    reset_global_state = args.reset_global_state
    remove_extra_optim_states = args.remove_extra_optim_states
    reset_optim_states = args.reset_optim_states
    num_workers = args.num_workers
    # Enumerate all *model_state.pt files in ds_checkpoint_dir
    model_state_paths = get_model_state_paths(ds_checkpoint_dir)
    assert (
        len(model_state_paths) == world_size
    ), f"Expected {world_size} model state paths, found {len(model_state_paths)}"
    print(f"Found {len(model_state_paths)} model state paths", flush=True)

    optim_state_paths = get_optim_state_paths(ds_checkpoint_dir)
    print(f"Found {len(optim_state_paths)} optim state paths", flush=True)
    assert (
        len(optim_state_paths) == world_size
    ), f"Expected {world_size} optim state paths, found {len(optim_state_paths)}"

    model_paths_to_update = []
    optim_paths_to_update = []
    for i in range(len(model_state_paths)):
        model_state_path = model_state_paths[i]
        optim_state_path = optim_state_paths[i]
        model_mp_rank, model_dp_rank = get_model_mp_dp_ranks(model_state_path)
        optim_mp_rank, optim_dp_rank = get_optim_mp_dp_ranks(optim_state_path)
        assert model_mp_rank == optim_mp_rank, f"{os.getpid()}: Model and optim mp_rank mismatch {os.path.basename(model_state_path)} {os.path.basename(optim_state_path)}: {model_mp_rank} != {optim_mp_rank}"
        assert model_dp_rank == optim_dp_rank, f"{os.getpid()}: Model and optim dp_rank mismatch {os.path.basename(model_state_path)} {os.path.basename(optim_state_path)}: {model_dp_rank} != {optim_dp_rank}"
        if model_mp_rank >= start_mp_rank and model_mp_rank <= end_mp_rank:
            model_paths_to_update.append(model_state_path)
            optim_paths_to_update.append(optim_state_path)

    print(f"{os.getpid()}: Updating {len(model_paths_to_update)} model states and {len(optim_paths_to_update)} optim states", flush=True)
    # Process each model state path
    start_time = time.time()

    if not args.skip_model_states:
        update_model_states(
            num_groups=num_groups,
            mp_size=mp_size,
            source_seq_len=source_seq_len,
            target_seq_len=target_seq_len,
            model_state_paths=model_paths_to_update,
            source_model_config=source_model_config,
            target_model_config=target_model_config,
            output_dir=output_dir,
            remove_extra_states=remove_extra_model_states,
            reset_global_state=reset_global_state,
            verbose=verbose,
            num_workers=args.num_workers,
            start_mp_rank=start_mp_rank,
            end_mp_rank=end_mp_rank,
        )
        duration = time.time() - start_time
        print(f"Finished updating model states, took: {duration:.2f} seconds", flush=True)

    if not args.skip_optim_states:
        start_time = time.time()
        update_optimizer_states(
            num_groups=num_groups,
            mp_size=mp_size,
            dp_size=dp_size,
            source_seq_len=source_seq_len,
            target_seq_len=target_seq_len,
            model_state_paths=model_paths_to_update,
            optim_state_paths=optim_paths_to_update,
            source_model_config=source_model_config,
            target_model_config=target_model_config,
            output_dir=output_dir,
            remove_extra_optim_states=remove_extra_optim_states,
            reset_optim_states=reset_optim_states,
            verbose=verbose,
            num_workers=num_workers,
            start_mp_rank=start_mp_rank,
            end_mp_rank=end_mp_rank,
        )
        duration = time.time() - start_time
        print(f"Finished updating optimizer states, took: {duration:.2f} seconds", flush=True)


def extend_zero3_checkpoint(
    args,
    source_model_config,
    target_model_config,
    checkpoint_dir,
    output_dir,
    verbose: bool = False,
):
    tag = args.tag
    source_seq_len = args.source_seq_len
    target_seq_len = args.target_seq_len
    num_groups = args.num_groups

    if tag is None:
        latest_path = os.path.join(checkpoint_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
            print(f"Using tag '{tag}' from 'latest' file")
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

    print(
        f"Extending checkpoint {ds_checkpoint_dir} from {source_seq_len} to {target_seq_len} using {num_groups} filter groups for explicit single decay filters", flush=True
    )

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    # Default to global_step0
    output_dir = os.path.join(output_dir, tag)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    extend_filters(
        args, ds_checkpoint_dir, source_model_config, target_model_config, output_dir, verbose=verbose
    )

def normalize_model_config(model_config: dict) -> dict:
    config = {}
    for k in model_config.keys():
        config[k.replace("-", "_")] = model_config[k]
    return config


def parse_operator_config(model_config: dict) -> dict:
    operator_counts = defaultdict(int)
    for op_type, op_count in model_config["operator_config"]:
        assert len(op_type) == 1, f"Expected single character operator type, found {op_type}"
        op_type = op_type[0]
        operator_counts[op_type] += op_count
    return operator_counts


def load_model_config(model_config_path: str) -> dict:
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    return normalize_model_config(model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir", type=str, help="path to the desired checkpoint folder, e.g., path/checkpoint-12"
    )
    parser.add_argument(
        "--source_model_config",
        type=str,
        help="Path to the source model config file",
    )
    parser.add_argument(
        "--target_model_config",
        type=str,
        help="Path to the target model config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extended_checkpoints",
        help="directory to the pytorch fp32 state_dict output files" "(e.g. path/checkpoint-12-output/)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP shards")
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=None,
        help="checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="enable debug")
    parser.add_argument("--dp_size", default=256, type=int, help="Data parallel size of source checkpoint")
    parser.add_argument("--mp_size", default=8, type=int, help="Model parallel size of source checkpoint")
    parser.add_argument(
        "--source_seq_len", default=None, choices=SOURCE_SEQ_LEN, type=int, help="Source sequence length"
    )
    parser.add_argument(
        "--target_seq_len", default=None, choices=TARGET_SEQ_LENS, type=int, help="Target sequence length"
    )
    parser.add_argument("--num_groups", default=None, type=int, help="Number of groups")
    parser.add_argument(
        "--keep_extra_model_states",
        action="store_true",
        help="Keep extra model states (data loading, ds_config, etc.)",
    )
    parser.add_argument(
        "--keep_global_state",
        action="store_true",
        help="Keep global state (global_steps, global_samples, skipped_steps, etc.)",
    )
    parser.add_argument(
        "--remove_extra_optim_states", action="store_true", help="Remove extra optimizer states"
    )
    parser.add_argument("--keep_optim_states", action="store_true", help="Keep optimizer states")
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of workers to use for processing"
    )
    parser.add_argument("--skip_model_states", action="store_true", help="Skip model states")
    parser.add_argument("--skip_optim_states", action="store_true", help="Skip optimizer states")
    parser.add_argument("--start_mp_rank", type=int, default=None, help="Start MP rank")
    parser.add_argument("--end_mp_rank", type=int, default=None, help="End MP rank")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()

    args.world_size = args.dp_size * args.mp_size
    source_model_config = load_model_config(args.source_model_config)
    target_model_config = load_model_config(args.target_model_config)

    num_groups = source_model_config["num_groups_hyena_medium"]
    assert (
        num_groups == target_model_config["num_groups_hyena_medium"]
    ), f"num groups should be same for source and target model config, found {num_groups} and {args.num_groups}"

    args.source_seq_len = source_model_config["seq_length"]
    args.target_seq_len = target_model_config["seq_length"]
    args.num_groups = num_groups

    args.remove_extra_model_states = not args.keep_extra_model_states
    args.reset_global_state = not args.keep_global_state
    args.reset_optim_states = not args.keep_optim_states

    if args.num_workers <= 0:
        args.num_workers = multiprocessing.cpu_count()

    if args.start_mp_rank is None:
        args.start_mp_rank = 0
    if args.end_mp_rank is None:
        args.end_mp_rank = args.mp_size

    print(
        f"Extending checkpoint {args.checkpoint_dir} from {args.source_seq_len} to {args.target_seq_len} using {args.num_groups} filter groups for explicit single decay filters with {args.num_workers} workers", flush=True
    )
    print("Script args:", flush=True)
    for k, v in vars(args).items():
        print(f" {k}: {v}", flush=True)
    extend_zero3_checkpoint(
        args,
        source_model_config=source_model_config,
        target_model_config=target_model_config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )