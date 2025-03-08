
import argparse
import os
import time
from multiprocessing import Pool

import torch
from partition_lib import FP8_SHAPE, FP8_SHAPE_T, get_all_model_files, is_mlp_weight

CHECKPOINT_DIR = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero3/MP16DP128/padded/global_step0"

model_files = get_all_model_files(CHECKPOINT_DIR)

def update_model_state(model_file):
    start = time.time()
    print(f"{os.getpid()}: Updating {os.path.basename(model_file)}", flush=True)
    model_state = torch.load(model_file)
    param_shape_groups = model_state["param_shapes"]
    for param_shape_group in param_shape_groups:
        for name, p_shape in param_shape_group.items():
            if is_mlp_weight(name):
                assert isinstance(p_shape, tuple), f"{os.getpid()}: {name}: {type(p_shape)} is not a tuple"
                assert len(p_shape) == 2, f"{os.getpid()}: {name}: {p_shape} is not a tuple of length 2"
                if "w1" in name or "w2" in name:
                    assert p_shape[0] % FP8_SHAPE_T[0] == 0, f"{os.getpid()}: {name}: {p_shape} is not a multiple of {FP8_SHAPE_T[0]}"
                    assert p_shape[1] % FP8_SHAPE_T[1] == 0, f"{os.getpid()}: {name}: {p_shape} is not a multiple of {FP8_SHAPE_T[1]}"
                    param_shape_group[name] = torch.Size(p_shape)
                    print(f"{os.getpid()}: {name}: {p_shape} -> {param_shape_group[name]}")
                elif "w3" in name:
                    assert p_shape[0] % FP8_SHAPE[0] == 0, f"{os.getpid()}: {name}: {p_shape} is not a multiple of {FP8_SHAPE[0]}"
                    assert p_shape[1] % FP8_SHAPE[1] == 0, f"{os.getpid()}: {name}: {p_shape} is not a multiple of {FP8_SHAPE[1]}"
                    param_shape_group[name] = torch.Size(p_shape)
                    print(f"{os.getpid()}: {name}: {p_shape} -> {param_shape_group[name]}")

    # Make sure model_state is updated
    for param_shape_group in model_state["param_shapes"]:
        assert all(isinstance(p_shape, torch.Size) for p_shape in param_shape_group.values()), f"{os.getpid()}: {param_shape_group} is not a dictionary of torch.Size"
    print(f"{os.getpid()}: {os.path.basename(model_file)} updated in {time.time() - start:.2f} seconds", flush=True)
    torch.save(model_state, model_file)



def check_model_state(model_file):
    start = time.time()
    print(f"{os.getpid()}: Checking {os.path.basename(model_file)}", flush=True)
    model_state = torch.load(model_file)
    for param_shape_group in model_state["param_shapes"]:
        assert all(isinstance(p_shape, torch.Size) for p_shape in param_shape_group.values()), f"{os.getpid()}: {param_shape_group} is not a dictionary of torch.Size"
    print(f"{os.getpid()}: {os.path.basename(model_file)} checked", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    if args.update:
        with Pool(processes=4) as pool:
            pool.map(update_model_state, model_files)
    else:
        with Pool(processes=4) as pool:
            pool.map(check_model_state, model_files)

