"""
Overwrite train_data_token_index in model_states.pt files to skip to a target iteration.

NOTE: If running on NVIDIA must run this using same container with pinned packages.

Different torch / numpy versions can cause issues.
"""

import glob
import math
import multiprocessing
import os
import re
import time

import torch

TEST_CHECKPOINT_DIR = "test_checkpoints/global_step100"

CURRENT_ITERATION = 519600
CHECKPOINT_BASE="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
CHECKPOINT="40b-train-n256-v2/40b_train_v2/202410271619"

#CHECKPOINT="40b-train-n256-8K-backup"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, CHECKPOINT, f"global_step{CURRENT_ITERATION}")
GLOBAL_BATCH_SIZE_TOKENS = 16777216

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

def calc_target_token_index(target_iteration, batch_size_tokens=GLOBAL_BATCH_SIZE_TOKENS):
    return int(target_iteration * batch_size_tokens)

def print_target_token_indices(target_iterations=[525_000,530_000, 535_000, 540_000]):
    for target_iteration in target_iterations:
        TARGET_TOKEN_INDEX = calc_target_token_index(target_iteration)
        print(f"Target: {target_iteration:9} Target Token Index: {TARGET_TOKEN_INDEX:.2e}")

def overwrite_token_index(model_state_path, target_token_index, dryrun=True):
    model_state = torch.load(model_state_path)
    current_token_index = model_state['data_loading']['train_data_token_index']

    print(f"{os.getpid()}::overwriting {os.path.basename(model_state_path)}: {current_token_index:.1e} -> {target_token_index:.1e}", flush=True)
    if not dryrun:
        model_state['data_loading']['train_data_token_index'] = int(target_token_index)
        torch.save(model_state, model_state_path)

def check_overwrite(model_state_path, target_token_index):
    model_state = torch.load(model_state_path)
    current_token_index = model_state['data_loading']['train_data_token_index']
    if not current_token_index == target_token_index:
        print(f"{os.getpid()}::{os.path.basename(model_state_path)}::FAILED: {current_token_index:.1e} != {target_token_index:.1e}", flush=True)
        return False
    else:
        print(f"{os.getpid()}::{os.path.basename(model_state_path)}::PASSED: {current_token_index:.1e} == {target_token_index:.1e}", flush=True)
        return True

if __name__ == "__main__":
    DRYRUN = False
    TARGET_ITERATION = 530_000
    NUM_WORKERS = os.cpu_count()
#    CHECKPOINT_DIR = TEST_CHECKPOINT_DIR    

    target_index = calc_target_token_index(TARGET_ITERATION)
    print(f"Target: {TARGET_ITERATION:9} Target Token Index: {target_index:.1e}")
    
    model_state_paths = get_model_state_paths(CHECKPOINT_DIR)
    assert len(model_state_paths) == 2048

    start_time = time.time()
    if NUM_WORKERS > 1:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            pool.starmap(overwrite_token_index, [(path, target_index, DRYRUN) for path in model_state_paths])    
    else:
        for path in model_state_paths:
            overwrite_token_index(path, target_index, DRYRUN)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print("Checking overwrites...")
    start_time = time.time()
    if not DRYRUN:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:   
            passed = pool.starmap(check_overwrite, [(path, target_index) for path in model_state_paths])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    assert all(passed), "Not all overwrites passed"
