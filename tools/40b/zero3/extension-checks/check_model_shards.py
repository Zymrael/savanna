import os

import torch
from partition_lib import MODEL_KEY, PARAM_SHAPE_KEY, get_all_shard_files

DEVICE = "cpu"
SOURCE_SHARDS = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/8K/zero1/global_step516000"
TARGET_SHARDS = "repartitioned_checkpoints/8K/MP8DP4/zero1/global_step0"


def check_param_shapes(source_shard_state, target_shard_state):
    source_shapes = source_shard_state[PARAM_SHAPE_KEY]
    target_shapes = target_shard_state[PARAM_SHAPE_KEY]
    assert len(source_shapes) == len(target_shapes), f"Expected {len(source_shapes)} shards, found {len(target_shapes)} shards"
    assert source_shapes.keys() == target_shapes.keys(), f"Expected {source_shapes.keys()} shards, found {target_shapes.keys()} shards"
    for name, ref_shape in source_shapes.items():
        target_shape = target_shapes[name]
        assert ref_shape == target_shape, f"{name}: Expected {ref_shape}, found {target_shape}"

    print("Param shapes check passed!", flush=True)

def check_param_values(source_shard_state, target_shard_state, indent=2):
    source_model = source_shard_state[MODEL_KEY]
    target_model = target_shard_state[MODEL_KEY]

    assert source_model.keys() == target_model.keys(), f"{source_model.keys()} != {target_model.keys()}"
    for name in source_model.keys():
        source_param = source_model[name]
        target_param = target_model[name]
        if hasattr(source_param, "shape"):
            assert source_param.shape == target_param.shape, f"{name}: Expected {source_param.shape}, found {target_param.shape}"
            assert torch.equal(source_param, target_param), f"{name}: {source_param.view(-1)[:10]} != {target_param.view(-1)[:10]}"
        else:
            #Check extra states
            source_param.seek(0)
            source_extra_states = torch.load(source_param, map_location=DEVICE)
            target_param.seek(0)
            target_extra_states = torch.load(target_param, map_location=DEVICE)
            for k, v in source_extra_states.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, target_extra_states[k]), f"{name}: {k}: {v.view(-1)[:10]} != {target_extra_states[k].view(-1)[:10]}"
                else:
                    assert v == target_extra_states[k], f"{name}: {k}: {v} != {target_extra_states[k]}"
    
if __name__ == "__main__":
    source_shards = get_all_shard_files(SOURCE_SHARDS)
    target_shards = get_all_shard_files(TARGET_SHARDS)
    assert len(source_shards) == len(target_shards), f"Expected {len(source_shards)} shards, found {len(target_shards)} shards"
    assert len(source_shards) > 0, f"Expected at least 1 shard, found {len(source_shards)} shards"

    for source_shard, target_shard in zip(source_shards, target_shards):

        assert os.path.basename(source_shard) == os.path.basename(target_shard), f"Expected {os.path.basename(source_shard)} shards, found {os.path.basename(target_shard)} shards"
        print(f"Checking {os.path.basename(source_shard)}", flush=True)

        source_shard_state = torch.load(source_shard, map_location=DEVICE)
        target_shard_state = torch.load(target_shard, map_location=DEVICE)
        
        print(" -> Checking param shapes", flush=True)
        check_param_shapes(source_shard_state, target_shard_state)
        print(" -> Param shapes check passed!", flush=True)

        print(" -> Checking param values", flush=True)
        check_param_values(source_shard_state, target_shard_state)
        print(" -> Param values check passed!", flush=True)

    print("All shards match", flush=True)

    # Check param values