import argparse
import os
import re
import shutil
from pathlib import Path

import torch
import yaml
from common import (
    DEFAULT_CHECKPOINT_NAME,
    GLOBAL_STEP_PAT,
    copy_checkpoint_to_new_dir,
    get_layer_pattern,
)


def get_args():
    parser = argparse.ArgumentParser(description='Convert model checkpoint to interleaved', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_dir', type=str, required=True, help="Path to source checkpoint directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Name of output directory, full path will be {output_dir}/global_step{step} if source dir contains global_step{step}")
    parser.add_argument('--model_config', type=str, required=True, help="Path to model config")
    parser.add_argument('--checkpoint_name', type=str, default=DEFAULT_CHECKPOINT_NAME, help="Name of model checkpoint to load")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing checkpoint")
    parser.add_argument('--dry-run', action='store_true', help="Print out the changes that will be made without saving the new checkpoint")
    return parser.parse_args()


def reshape_to_interleaved(source_tensor, dim):
    assert source_tensor.shape[0] == 3 * dim, f"Source tensor shape {source_tensor.shape[0]} does not match expected shape {3 * dim}"
    temp_tensor = torch.nn.Parameter(torch.empty_like(source_tensor))
    
    for i in range(3):
        start = i * dim
        end = (i + 1) * dim
        temp_tensor[i::3, :] = source_tensor[start:end, :]

    return temp_tensor


def interleave_model_states(checkpoint_path, model_config):
    """
    Interleave dense proj and short conv weights in-place
    """
    
    dim = model_config["hidden_size"]
    layer_pattern = get_layer_pattern(model_config)
    state_dict = torch.load(checkpoint_path)

    with torch.no_grad():
        for i, layer_type in enumerate(layer_pattern):
            print(f"Layer {i+2} is {layer_type}")
            if layer_type == "flash_v2":
                print(f" -> Skipping layer {layer_type}, layer{i+2}")
            else:
                print(f"-> Reshaping layer {layer_type}, layer{i+2}")
                dense_pat = f"sequential.{i+2}.mixer.dense_projection.weight"
                assert dense_pat in state_dict['module'], f"Pattern {dense_pat} not found in checkpoint"
                print(f"   -> matched {dense_pat}")
                dense_projection = state_dict['module'][dense_pat]
                # print(dense_projection.shape)
                state_dict['module'][dense_pat] = reshape_to_interleaved(dense_projection, dim=dim)

                short_conv_pat = f'sequential.{i+2}.mixer.hyena_proj_conv.short_conv_weight'
                print(f"   -> matched {short_conv_pat}")
                assert short_conv_pat in state_dict['module'], f"Pattern {short_conv_pat} not found in checkpoint"
                short_conv_weight = state_dict['module'][short_conv_pat]
                state_dict['module'][short_conv_pat] = reshape_to_interleaved(short_conv_weight, dim=dim)
                # print(short_conv_weight.shape)
    return state_dict


def setup_dirs(args):

    # Check if source checkpoint exists
    source_checkpoint = os.path.join(args.source_dir, args.checkpoint_name)
    assert os.path.exists(source_checkpoint), f"Source checkpoint {source_checkpoint} does not exist"


    # Match global_step{step} from source checkpoint
    step = re.search(GLOBAL_STEP_PAT, args.source_dir).group(1)
    output_dir = Path(args.output_dir)
    if step is not None:
        output_dir = output_dir / f"global_step{step}"
    else:
        print(f"WARNING: Could not find global_step in {args.source_dir}")
    
    output_path = output_dir / args.checkpoint_name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if output_path.exists() and not args.overwrite:
        raise ValueError(f"{output_path} already exists. Specify --overwrite to continue.")
        
    copy_checkpoint_to_new_dir(source_checkpoint, output_path, dry_run=args.dry_run)
    return source_checkpoint, output_path

def main(args):
    
    source_checkpoint, output_checkpoint = setup_dirs(args)
    print(f"Interleaving source from {source_checkpoint} -> {output_checkpoint}")
    
    print(f"Loading model config from {args.model_config}...")
    with open(args.model_config, "r") as file:
        model_config = yaml.safe_load(file)

    # Interleaved model states in-place in output checkpoint
    print("Interleaving model states...")
    state_dict = interleave_model_states(output_checkpoint, model_config)

    if not args.dry_run:
        print(f"Saving interleaved checkpoint to {output_checkpoint}...")
        torch.save(state_dict, output_checkpoint)
    print("Done!")

if __name__ == "__main__":
    args = get_args()
    main(args)