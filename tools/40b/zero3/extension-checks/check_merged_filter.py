import glob
import os

import torch

SOURCE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-512K/zero1/8K/global_step516000"
TARGET_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/256K/zero1/global_step0"
FILTER_PATTERN = "mixer.filter.h"
FILTER_LEN = 128
SOURCE_SEQ_LEN = 8 * 2 ** 10
TARGET_SEQ_LEN = 256 * 2 ** 10

def get_filter_stats(filter):
    return filter.mean(), filter.min(), filter.max()

if __name__ == "__main__":
    source_shards = sorted(glob.glob(f"{SOURCE_DIR}/mp*.pt"))
    target_shards = sorted(glob.glob(f"{TARGET_DIR}/mp*.pt"))

    assert len(source_shards) == len(target_shards), f"{len(source_shards)=} != {len(target_shards)=}"
    assert len(source_shards) == 8

    for shard1, shard2 in zip(source_shards, target_shards):
        shard_name = os.path.basename(shard1)
        assert os.path.basename(shard2) == shard_name
        print(f"Checking {shard_name}", flush=True)
        model_1 = torch.load(shard1)['module']
        model_2 = torch.load(shard2)['module']

        assert model_1.keys() == model_2.keys()
        
        for name, param in model_1.items():
            if FILTER_PATTERN in name:
                h1 = param[:,:FILTER_LEN]
                h2 = model_2[name][:,:FILTER_LEN]
                ref_mean, ref_min, ref_max = get_filter_stats(h1)
                new_mean, new_min, new_max = get_filter_stats(h2)
                print(f"{name}:filter.h {ref_mean:.4f}/{new_mean:.4f} | {ref_min:.4f}/{new_min:.4f} | {ref_max:.4f}/{new_max:.4f}")
                
                # h1_remainder = param[:,FILTER_LEN:]
                # h2_remainder = model_2[name][:,FILTER_LEN:SOURCE_SEQ_LEN]
                # ref_mean, ref_min, ref_max = get_filter_stats(h1_remainder)
                # new_mean, new_min, new_max = get_filter_stats(h2_remainder)
                # print(f"{name}:filter-remainder: {ref_mean:.4f}/{new_mean:.4f} | {ref_min:.4f}/{new_min:.4f} | {ref_max:.4f}/{new_max:.4f}")
                
                # h2_rest = model_2[name][:,SOURCE_SEQ_LEN:]
                # new_mean, new_min, new_max = get_filter_stats(h2_rest)
                # print(f"{name}:rest: {new_mean:.4f} | {new_min:.4f} | {new_max:.4f}")
                # print()
        
                assert torch.allclose(h1, h2)
                
        print("-" * 100)
