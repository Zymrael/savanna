import os

import torch

from savanna import print_rank_0

DEBUG_TENSOR_SAVE_DIR = None

ABBREV_MAP = {
    "model_parallel_size": "mp",
    "context_parallel_size": "cp",
    "use_cp_hyena": "cp_hyena",
    "use_cp_ring": "ring_attn",
    "use_cp_flash_te": "flash_te",
}
def save_tensor_hook(name, module, global_config, config_keys=["model_parallel_size", "context_parallel_size", "use_cp_hyena", "use_cp_ring", "use_cp_flash_te"]):
    global DEBUG_TENSOR_SAVE_DIR

    rank = torch.distributed.get_rank()

    debug_dir = global_config.debug_dir
    assert all([hasattr(global_config, k) for k in config_keys])

    config_str = "_".join([f"{ABBREV_MAP[k] if k in ABBREV_MAP else k}={v}" for k, v in zip(config_keys, [getattr(global_config, k) for k in config_keys])])
    parent_dir = os.path.join(debug_dir, config_str)
    global_config.debug_save_dir = parent_dir

    save_dir = os.path.join(parent_dir, name) 
    print_rank_0(f"DEBUG::FORWARD_HOOK: Saving {name} to {save_dir}")
    if rank == 0:
        import yaml
        os.makedirs(save_dir, exist_ok=True)
        yaml.dump(global_config, open(f"{parent_dir}/global_config.yml", "w"))

    def _save_tensor_hook(module, inputs, output):
        if not rank == 0:
            return
    
        print(f"DEBUG::FORWARD_HOOK: Saving {name} to {save_dir}", flush=True)
        for i, input in enumerate(inputs):
            if isinstance(input, torch.Tensor):
                print(f" -> Saving input {i}")
                torch.save(input.detach().clone(), f"{save_dir}/inputs{i}.pt")
            else:
                print(f" -> Skipping input {i}: non-tensor type {type(input)}", flush=True)
        if isinstance(output, torch.Tensor):
            print(" -> Saving output", flush=True)
            torch.save(output.clone().detach(), f"{save_dir}/output.pt")
        elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        print(f" -> Saving output {i}")
                        torch.save(o.clone().detach(), f"{save_dir}/output{i}.pt")
        else:
            print(f" -> Skipping output: non-tensor type {type(output)}", flush=True)

    module_name = module.__class__.__name__
    modules_to_skip = {"ParallelBlockPipe", "Lambda"}
    if module_name in modules_to_skip:
        print_rank_0(f"DEBUG::FORWARD_HOOK: Skipping forward hook for {module_name}")
        return
    print_rank_0(f"DEBUG::FORWARD_HOOK: Setting up forward hook for {module_name}:{name} {len(module._modules)}")
    return module.register_forward_hook(_save_tensor_hook)
