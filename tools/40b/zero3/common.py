import os
import re
import shutil
from pathlib import Path

import torch

#  --------------- Constants --------------- #
DEFAULT_CHECKPOINT_NAME = "mp_rank_00_model_states.pt"
GLOBAL_STEP_PAT = re.compile(r'global_step(\d+)')

DENSE_PROJ_PAT = "mixer.dense_projection.weight"
SHORT_CONV_PROJ_PAT = "mixer.hyena_proj_conv.short_conv_weight"

DEFAULT_PARAM_PATTERN = r'sequential\.\d+\.(.+)'
ATTENTION_LAYER_TYPES = ["flash_v2", "ring", "flash_te"]

# --------------- Utilities --------------- #

# Copy all files from src_dir to dest_dir
def copy_checkpoints_to_new_dir(src_dir, dest_dir):
    print(f"Copying checkpoints from {src_dir} to {dest_dir}")
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

# Only copy model states, not optimizer states
def copy_checkpoint_to_new_dir(source_checkpoint, destination_checkpoint, dry_run=False):
    print(f"Copying checkpoint from {source_checkpoint} to {destination_checkpoint}")
    if not dry_run:
        shutil.copy(source_checkpoint, destination_checkpoint)

def remove_old_model_states(output_path):
    assert output_path.exists()
    print(f"Removing old model states {output_path}")
    os.remove(output_path)


def load_checkpoint(checkpoint_path):
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    state_dict = torch.load(checkpoint_path)
    assert 'module' in state_dict
    return state_dict['module']

def load_state_dict(source_dir, checkpoint_name):
    source_dir = Path(source_dir) if not isinstance(source_dir, Path) else source_dir

    model_files = list(source_dir.glob("*.pt"))
    assert (
        len(model_files) == 1
    ), f"Expected one model file, found {len(model_files)} {model_files}"
    model_file = model_files[0]
    assert checkpoint_name in model_file.name
    print(f"Loading model state {model_file}...")
    model_state_dict = torch.load(model_file)

    return model_state_dict

def load_model_dict(args):
    model_state_dict = load_state_dict(args.source_dir, args.checkpoint_name)
    assert "module" in model_state_dict
    model_dict = model_state_dict["module"]

    return model_dict

# --------------- CLI --------------- #

def parse_to_list(s):
    return s.split(",")

# --------------- Interleaving --------------- #
def get_layer_pattern(config):
    operator_config = config.get("operator-config")
    layer_pattern = [item[0][0] for item in operator_config]
    return layer_pattern

# --------------- Filter Extension --------------- #
EXPLICIT_FILTER_PATTERNS = ["mixer.mixer.filter.h", "mixer.mixer.filter.decay"]
IMPLICIT_FILTER_PATTERN = "mixer.mixer.filter.t"

EXPLICIT_SINGLE_DECAY_BUFFER = "mixer.mixer.filter.decay"
IMPLICIT_MODAL_BUFFER = "mixer.mixer.filter.t"

EXPLICIT_SINGLE_DECAY_PARAM = "mixer.mixer.filter.h"
