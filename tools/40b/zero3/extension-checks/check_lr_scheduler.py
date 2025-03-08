import argparse

import torch
from partition_lib import get_all_model_files, load_model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--target_config_path", type=str)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    model_files = get_all_model_files(checkpoint_dir)
    target_config_path = args.target_config_path
    target_model_config = load_model_config(target_config_path)

    for model_file in model_files:
        print(f"Checking {model_file}")
        model_state = torch.load(model_file, map_location="cpu")
        lr_scheduler = model_state["lr_scheduler"]
        
        assert lr_scheduler["min_lr"] == target_model_config["min_lr"], f"{model_file} min_lr {lr_scheduler['min_lr']} != {target_model_config['min_lr']}"
        assert lr_scheduler["start_lr"] == target_model_config["optimizer"]["params"]["lr"], f"{model_file} start_lr {lr_scheduler['start_lr']} != {target_model_config['optimizer']['params']['lr']}"
        assert lr_scheduler["num_iters"] == 0, f"{model_file} num_iters {lr_scheduler['num_iters']} != 0"
        assert lr_scheduler["end_iter"] == target_model_config["train_iters"], f"{model_file} end_iter {lr_scheduler['end_iter']} != {target_model_config['train_iters']}"
        assert lr_scheduler["decay_style"] == target_model_config["lr_decay_style"], f"{model_file} decay_style {lr_scheduler['decay_style']} != {target_model_config['lr_decay_style']}"
        assert lr_scheduler["warmup_iter"] == int(target_model_config["warmup"] * target_model_config["train_iters"]), f"{model_file} warmup_iter {lr_scheduler['warmup_iter']} != {int(target_model_config['warmup'] * target_model_config['train_iters'])}"
        
        print(f"{model_file} passed!")
