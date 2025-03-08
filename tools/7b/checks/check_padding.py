import argparse

import torch
from common import FP8_SHAPE, FP8_SHAPE_T, get_all_shard_files, is_mlp_weight


def check_param_shapes(param_shapes, model_file):
    failed_count = 0
    for name, shape in param_shapes.items():
        if "w1" in name or "w2" in name:
            if shape[0] % 16 != 0 and shape[1] % 8 != 0:
                print(f"{model_file} failed {name} {shape}", flush=True)
                failed_count += 1
        elif "w3" in name:
            if shape[1] % 16 != 0 and shape[0] % 8 != 0:
                print(f"{model_file} failed {name} {shape}", flush=True)
                failed_count += 1
    
    return failed_count

def check_params(model_state_dict):
    for param_name, param in model_state_dict.items():
        if is_mlp_weight(param_name):
            if "w1" in param_name or "w2" in param_name:
                assert param[param.shape[0]:, :].equal(torch.zeros_like(param[param.shape[0]:, :]))
                assert param[:, param.shape[1]:].equal(torch.zeros_like(param[:, param.shape[1]:]))
                assert param.shape[0] % FP8_SHAPE_T[0] == 0, f"Expected shape {param.shape} to be a multiple of {FP8_SHAPE_T} after padding, found {param.shape[0]}"
                assert param.shape[1] % FP8_SHAPE_T[1] == 0, f"Expected shape {param.shape} to be a multiple of {FP8_SHAPE_T} after padding, found {param.shape[1]}"        
            elif "w3" in param_name:
                assert param[param.shape[0]:, :].equal(torch.zeros_like(param[param.shape[0]:, :]))
                assert param[:, param.shape[1]:].equal(torch.zeros_like(param[:, param.shape[1]:]))
                assert param.shape[0] % FP8_SHAPE[0] == 0, f"Expected shape {param.shape} to be a multiple of {FP8_SHAPE} after padding, found {param.shape[0]}"
                assert param.shape[1] % FP8_SHAPE[1] == 0, f"Expected shape {param.shape} to be a multiple of {FP8_SHAPE} after padding, found {param.shape[1]}"
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    args = parser.parse_args()

    model_files = get_all_shard_files(args.checkpoint_dir)
    assert len(model_files) > 0, f"Expected at least one shard file in {args.checkpoint_dir}"
    
    for model_file in model_files:

        print(f"Checking {model_file}", flush=True)
        model_state = torch.load(model_file, map_location="cpu")
        
        param_shapes = model_state['param_shapes']
        model_state_dict = model_state['module']

        failed_count = check_param_shapes(param_shapes, model_file)
        assert failed_count == 0, f"Param shape check failed: {failed_count} params in {model_file}"

        check_params(model_state_dict)
        print(f"{model_file}: Passed!", flush=True)
    
    print("Done!")