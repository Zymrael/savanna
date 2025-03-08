import torch
from partition_lib import get_all_model_files, get_all_optim_files, load_model_config

checkpoint_dir = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero3/MP16DP64/padded/global_step0"
model_files = get_all_model_files(checkpoint_dir)
target_config_path = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-long-ext/configs/40b/model_configs/extension/128K/40b_128K_MP16_test.yml"
target_model_config = load_model_config(target_config_path)

for model_file in model_files:
    print(f"Updating {model_file}")
    model_state = torch.load(model_file, map_location="cpu")
    
    updated_lr_scheduler = {}
    updated_lr_scheduler["min_lr"] = target_model_config["min_lr"]
    updated_lr_scheduler["start_lr"] = target_model_config["optimizer"]["params"]["lr"]
    updated_lr_scheduler["num_iters"] = 0
    updated_lr_scheduler["end_iter"] = target_model_config["train_iters"]
    updated_lr_scheduler["decay_style"] = target_model_config["lr_decay_style"]
    updated_lr_scheduler["warmup_iter"] = int(target_model_config["warmup"] * target_model_config["train_iters"])

    model_state["lr_scheduler"] = updated_lr_scheduler
    torch.save(model_state, model_file)
