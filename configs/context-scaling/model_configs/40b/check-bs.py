import argparse
import glob
import os

import yaml

MBS_KEY = "train_micro_batch_size_per_gpu"
GAS_KEY = "gradient_accumulation_steps"
SEQ_LENGTH_KEY = "seq_length"
MP_KEY = "model_parallel_size"
CP_KEY = "context_parallel_size"

def calculate_global_bs(mbs, gas, seq_length, mp, cp, num_gpus):
    dp_size = num_gpus // (mp * cp)
    return mbs * gas * seq_length * dp_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-path", type=str, default="generated")
    parser.add_argument("--num-gpus", type=int, default=1024)
    args = parser.parse_args()

    configs = list(sorted(glob.glob(os.path.join(args.configs_path, "*.yml"))))
    ref_config_name = configs[0]
    with open(ref_config_name, "r") as f:
        ref_config = yaml.load(f, Loader=yaml.FullLoader)
    
    ref_bs = calculate_global_bs(ref_config[MBS_KEY], ref_config[GAS_KEY], ref_config[SEQ_LENGTH_KEY], ref_config[MP_KEY], ref_config[CP_KEY], args.num_gpus)
    print(f"Reference config {os.path.basename(ref_config_name)}: {ref_bs}")

    for config_path in configs:
        print(f"Checking {os.path.basename(config_path)}...", end="")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        bs = calculate_global_bs(config[MBS_KEY], config[GAS_KEY], config[SEQ_LENGTH_KEY], config[MP_KEY], config[CP_KEY], args.num_gpus)
        assert bs == ref_bs, f"Global BS mismatch: {bs} != {ref_bs}"
        print("passed!")
