import argparse
from pathlib import Path

import torch
import yaml
from common import (
    DEFAULT_CHECKPOINT_NAME,
    DENSE_PROJ_PAT,
    SHORT_CONV_PROJ_PAT,
    get_layer_pattern,
    load_checkpoint,
)

DEFAULT_DIR_BASE = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
LEGACY_CHECKPOINT_DIR = f"{DEFAULT_DIR_BASE}/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/global_step500000"
MODEL_CONFIG = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/configs/7b-context-ext/model_configs/7b_stripedhyena2_base_4M_32k.yml"
INTERLEAVED_CHECKPOINT_DIR = f'{DEFAULT_DIR_BASE}/7b_evo2_conversion/interleaved/global_step500000/'


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

def check_model_states(args):
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    hidden_size = model_config['hidden_size']
    
    legacy_checkpoint = args.source_dir / args.checkpoint_name
    interleaved_checkpoint = args.output_dir / args.checkpoint_name
    legacy_model = load_checkpoint(legacy_checkpoint)
    interleaved_model = load_checkpoint(interleaved_checkpoint)
    layer_pattern = get_layer_pattern(model_config)
    
    for i, layer_type in enumerate(layer_pattern):
        layer_prefix = f"{layer_type}:layer{i+2}"
        print(f"-> Checking layer {layer_prefix}")
        
        # Case 1: flash_v2 layers should not be interleaved
        if layer_type == "flash_v2":
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
            
            passed = check_interleaved(legacy_dense_projection, updated_dense_projection, hidden_size)
            assert passed
            print(f" -> {layer_prefix} dense projection passed!")

            short_conv_pat = f'sequential.{i+2}.mixer.hyena_proj_conv.short_conv_weight'
            
            assert short_conv_pat in legacy_model, f"Pattern {short_conv_pat} not found in legacy checkpoint"
            assert short_conv_pat in interleaved_model, f"Pattern {short_conv_pat} not found in updated checkpoint"
            
            short_conv_weight = legacy_model[short_conv_pat]
            updated_short_conv_weight = interleaved_model[short_conv_pat]
            
            passed = check_interleaved(short_conv_weight, updated_short_conv_weight, hidden_size)
            assert passed
            print(f" -> {layer_prefix} short conv projection passed!")
    
    print("All checks passed!")

def get_args():
    parser = argparse.ArgumentParser(description='Check checkpointed model state has been properly interleaved', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_dir', type=Path, default=LEGACY_CHECKPOINT_DIR, help="Path to model checkpoint directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to updated, interleaved model checkpoint")
    parser.add_argument('--model_config', type=str, default=MODEL_CONFIG, help="Path to model config")
    parser.add_argument('--checkpoint_name', type=str, default=DEFAULT_CHECKPOINT_NAME, help="Name of model checkpoint to load")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    check_model_states(args)


