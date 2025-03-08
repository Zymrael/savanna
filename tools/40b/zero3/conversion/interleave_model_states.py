import argparse
import os
import re
import shutil
from pathlib import Path

import torch
import yaml
from common import (
    ATTENTION_LAYER_TYPES,
    GLOBAL_STEP_PAT,
    copy_checkpoint_to_new_dir,
    get_layer_pattern,
    load_checkpoint,
)
from partition_lib import get_all_shard_files


def get_args():
    parser = argparse.ArgumentParser(description='Convert model checkpoint to interleaved', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_dir', type=str, required=True, help="Path to source checkpoint directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Name of output directory, full path will be {output_dir}/global_step{step} if source dir contains global_step{step}")
    parser.add_argument('--model_config', type=str, required=True, help="Path to model config")
    parser.add_argument('--mp_size', type=int, default=8, help="MP size of source checkpoint")
    parser.add_argument('--dry-run', action='store_true', help="Print out the changes that will be made without saving the new checkpoint")
    return parser.parse_args()



def check_interleaved(original: torch.Tensor, interleaved: torch.Tensor, dim: int):    
    passed = False
    for i in range(3):
        start = i * dim
        end = (i + 1) * dim
        expected = original[start:end, :]
        actual = interleaved[i::3, :]
        diff = (expected - actual).abs().max()
        if not torch.equal(expected, actual):
            print(f"   -> Interleaved at idx {i} failed: {diff:.4f}")
            passed = False      
        else:
            print(f"   -> Interleaved at idx {i} passed: {diff:.4f}")
            passed = True
    return passed


def check_model_states(source_checkpoint, interleaved_checkpoint, model_config, mp_size):
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    hidden_size = model_config['hidden_size']
    hidden_size_per_rank = hidden_size // mp_size
    # legacy_checkpoint = args.source_dir / args.checkpoint_name
    # interleaved_checkpoint = args.output_dir / args.checkpoint_name
    source_checkpoint = Path(source_checkpoint)
    interleaved_checkpoint = Path(interleaved_checkpoint)
    legacy_model = load_checkpoint(source_checkpoint)
    interleaved_model = load_checkpoint(interleaved_checkpoint)
    layer_pattern = get_layer_pattern(model_config)
    
    for i, layer_type in enumerate(layer_pattern):
        layer_prefix = f"{layer_type}:layer{i+2}"
        print(f"-> Checking layer {layer_prefix}")
        
        # Case 1: attention layers should not be interleaved
        if layer_type in ATTENTION_LAYER_TYPES:
            dense_pat = f"sequential.{i+2}.mixer.dense_projection.weight"
            if dense_pat in legacy_model:
                assert dense_pat in interleaved_model, f"Pattern {dense_pat} not found in updated checkpoint"
                
                legacy_dense_projection = legacy_model[dense_pat]
                updated_dense_projection = interleaved_model[dense_pat]
                assert legacy_dense_projection.equal(updated_dense_projection)
            
            short_conv_pat = f'sequential.{i+2}.mixer.hyena_proj_conv.short_conv_weight'
            if short_conv_pat in legacy_model:
                assert short_conv_pat in interleaved_model, f"Pattern {short_conv_pat} not found in updated checkpoint"
                
                legacy_short_conv_weight = legacy_model[short_conv_pat]
                updated_short_conv_weight = interleaved_model[short_conv_pat]
                assert legacy_short_conv_weight.equal(updated_short_conv_weight)
            print(f"  -> {layer_type}:layer{i+2} passed!")
        # Case 2: dense proj and short conv layers should be interleaved
        else:
            print(f"-> Checking layer {layer_type}, layer{i+2}")
            dense_pat = f"sequential.{i+2}.mixer.dense_projection.weight"
            assert dense_pat in legacy_model, f"Pattern {dense_pat} not found in legacy checkpoint"

            assert dense_pat in interleaved_model, f"Pattern {dense_pat} not found in updated checkpoint"
            
            legacy_dense_projection = legacy_model[dense_pat]
            updated_dense_projection = interleaved_model[dense_pat]
            
            passed = check_interleaved(legacy_dense_projection, updated_dense_projection, hidden_size_per_rank)
            assert passed
            print(f" -> {layer_prefix} dense projection passed!")

            short_conv_pat = f'sequential.{i+2}.mixer.hyena_proj_conv.short_conv_weight'
            
            assert short_conv_pat in legacy_model, f"Pattern {short_conv_pat} not found in legacy checkpoint"
            assert short_conv_pat in interleaved_model, f"Pattern {short_conv_pat} not found in updated checkpoint"
            
            short_conv_weight = legacy_model[short_conv_pat]
            updated_short_conv_weight = interleaved_model[short_conv_pat]
            
            passed = check_interleaved(short_conv_weight, updated_short_conv_weight, hidden_size_per_rank)
            assert passed
            print(f" -> {layer_prefix} short conv projection passed!")
    
    print("All checks passed!")

def reshape_to_interleaved(source_tensor, dim, mp_size):
    dim_per_rank = dim // mp_size
    assert source_tensor.shape[0] == 3 * dim_per_rank, f"Source tensor shape {source_tensor.shape[0]} does not match expected shape {3 * dim_per_rank}"
    temp_tensor = torch.nn.Parameter(torch.empty_like(source_tensor))
    
    for i in range(3):
        start = i * dim_per_rank
        end = (i + 1) * dim_per_rank
        temp_tensor[i::3, :] = source_tensor[start:end, :]

    return temp_tensor


def interleave_model_states(checkpoint_path, model_config, mp_size):
    """
    Interleave dense proj and short conv weights in-place
    """
    
    dim = model_config["hidden_size"]
    layer_pattern = get_layer_pattern(model_config)
    state_dict = torch.load(checkpoint_path)

    with torch.no_grad():
        for i, layer_type in enumerate(layer_pattern):
            print(f"Layer {i+2} is {layer_type}")
            if layer_type in ATTENTION_LAYER_TYPES:
                print(f" -> Skipping layer {layer_type}, layer{i+2}")
            else:
                print(f"-> Reshaping layer {layer_type}, layer{i+2}")
                dense_pat = f"sequential.{i+2}.mixer.dense_projection.weight"
                assert dense_pat in state_dict['module'], f"Pattern {dense_pat} not found in checkpoint"
                print(f"   -> matched {dense_pat}")
                dense_projection = state_dict['module'][dense_pat]
                # print(dense_projection.shape)
                state_dict['module'][dense_pat] = reshape_to_interleaved(dense_projection, dim=dim, mp_size=mp_size)

                short_conv_pat = f'sequential.{i+2}.mixer.hyena_proj_conv.short_conv_weight'
                print(f"   -> matched {short_conv_pat}")
                assert short_conv_pat in state_dict['module'], f"Pattern {short_conv_pat} not found in checkpoint"
                short_conv_weight = state_dict['module'][short_conv_pat]
                state_dict['module'][short_conv_pat] = reshape_to_interleaved(short_conv_weight, dim=dim, mp_size=mp_size)
                # print(short_conv_weight.shape)
    return state_dict


def setup_dirs(source_dir, output_dir, mp_size):

    # Check if source checkpoint exists
    source_checkpoints = get_all_shard_files(source_dir)
    assert len(source_checkpoints) == mp_size, f"Expected {mp_size} source checkpoints, found {len(source_checkpoints)}"

    # Match global_step{step} from source checkpoint
    step = re.search(GLOBAL_STEP_PAT, source_dir).group(1)
    output_dir = Path(output_dir)
    if step is not None:
        output_dir = output_dir / f"global_step{step}"
    else:
        print(f"WARNING: Could not find global_step in {args.source_dir}")
    
    output_paths = [output_dir / os.path.basename(chkpt) for chkpt in source_checkpoints]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for source_checkpoint, output_path in zip(source_checkpoints, output_paths):
        if not output_path.exists():    
            copy_checkpoint_to_new_dir(source_checkpoint, output_path, dry_run=args.dry_run)
        else:
            print(f"Skipping {output_path} since it already exists, updating in-place")
    return source_checkpoints, output_paths

def main(args):
    
    source_checkpoints, output_checkpoints = setup_dirs(args.source_dir, args.output_dir, args.mp_size)
    print(f"Interleaving source from {source_checkpoints} -> {output_checkpoints}")
    
    print(f"Loading model config from {args.model_config}...")
    with open(args.model_config, "r") as file:
        model_config = yaml.safe_load(file)

    # Interleaved model states in-place in output checkpoint
    print("Interleaving model states...")
    for output_checkpoint in output_checkpoints:
        state_dict = interleave_model_states(output_checkpoint, model_config, mp_size=args.mp_size)

        if not args.dry_run:
            print(f"Saving interleaved checkpoint to {output_checkpoint}...")
            torch.save(state_dict, output_checkpoint)
    
    for source_checkpoint, output_checkpoint in zip(source_checkpoints, output_checkpoints):
        check_model_states(source_checkpoint, output_checkpoint, model_config, mp_size=args.mp_size)

    print(f"Done! Outputs written to {output_checkpoints}")

if __name__ == "__main__":
    args = get_args()
    main(args)