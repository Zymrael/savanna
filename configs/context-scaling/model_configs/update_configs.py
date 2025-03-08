import argparse
import glob
import os
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

CONFIGS_TO_EXCLUDE = []

KEYS_TO_DELETE = ["load", 
                  "num_workers",
                  "do_per_ds_valid",
                  "eval_per_ds_interval",
                  "eval_per_ds_iters",
                  "checkpoint-stores",
                  "finetune",
                  "use_checkpoint_lr_scheduler",
                  "use_checkpoint_num_samples"]
KEYS_TO_INSERT = {
    "use_cp_flash_te": True,
    "te_attn_backend": "FLASH",
}

CONVERTERS = {
    # "hidden_size": 4096,
    # "num_groups_hyena": 4096,
    # "num_groups_hyena_medium": 4096,
    # "num_groups_hyena_short": 4096,
    # "num_groups_hyena_mlp": 4096,
    # "checkpoint-factor": 5000,
    # "use_checkpoint_lr_scheduler": False,
    # "use_checkpoint_num_samples": False,
}

def convert_single(source_config_path, target_config_dir, additional_converters=None, output_name=None):
    print(f"Converting {source_config_path}...")
    with open(source_config_path, "r") as f:
        config = yaml.load(f)

    for key, value in CONVERTERS.items():
        prev_value = config.get(key, None)
        print(f"  changing {key} from {prev_value} to {value}")
        config[key] = value

    for key in KEYS_TO_DELETE:
        if key in config:
            del config[key]

    for key, value in KEYS_TO_INSERT.items():
        config[key] = value

    if additional_converters:
        for key, value in additional_converters.items():
            prev_value = config.get(key, None)
            print(f"  changing {key} from {prev_value} to {value}")
            config[key] = value

    output_name = output_name or os.path.basename(source_config_path)
    output_path = os.path.join(target_config_dir, output_name)
    with open(output_path, "w") as f:
        yaml.dump(config, f)

    print(f"Config saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_config_dir", default="7b/updated")
    parser.add_argument("--target_config_dir", default="7b/updated")
    args = parser.parse_args()

    if not os.path.exists(args.target_config_dir):
        os.makedirs(args.target_config_dir, exist_ok=True)

    configs_to_convert = list(filter(lambda p: not any(pat in p for pat in CONFIGS_TO_EXCLUDE), glob.glob(os.path.join(args.source_config_dir, "*.yml"))))
    for config_path in configs_to_convert:
        convert_single(config_path, args.target_config_dir)
