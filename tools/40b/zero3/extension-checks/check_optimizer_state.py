import argparse

import torch
from partition_lib import get_all_optim_files, load_model_config

OPTIMIZER_STATE_KEY = "optimizer_state_dict"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--target_config_path", type=str)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    optim_state_files = get_all_optim_files(checkpoint_dir)
    target_config_path = args.target_config_path
    target_model_config = load_model_config(target_config_path)

    for optim_state_file in optim_state_files:
        print(f"Checking {optim_state_file}", flush=True)
        optim_state = torch.load(optim_state_file, map_location="cpu")
        
        optimizer_state = optim_state[OPTIMIZER_STATE_KEY][OPTIMIZER_STATE_KEY]
        print(f"Number of param groups: {len(optimizer_state['param_groups'])}", flush=True)
        for param_group in optimizer_state["param_groups"]:
            print(f"param_group: {param_group}", flush=True)
            assert param_group["step"] == 0, f"Expected step to be 0, found {param_group['step']}"
            assert (
                param_group["lr"] == target_model_config["optimizer"]["params"]["lr"]
            ), f"Expected lr to be {target_model_config['optimizer']['params']['lr']}, found {param_group['lr']}"
            assert (
                param_group["betas"] == target_model_config["optimizer"]["params"]["betas"]
            ), f"Expected betas to be {target_model_config['optimizer']['params']['betas']}, found {param_group['betas']}"
            assert (
                param_group["eps"] == target_model_config["optimizer"]["params"]["eps"]
            ), f"Expected eps to be {target_model_config['optimizer']['params']['eps']}, found {param_group['eps']}"
        print(f"{optim_state_file} passed", flush=True)
