import glob
import os
import re


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
