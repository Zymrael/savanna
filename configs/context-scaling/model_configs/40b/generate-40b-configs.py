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
                  "use_checkpoint_num_samples",
                  "recycle_events",
                  "async_save",
                  "interleave_projections",
                  "iteration",
                  "warmstart",
                  "debug_print",
                  "save",
                ]

KEYS_TO_INSERT = {
    "use_cp_flash_te": True,
    "te_attn_backend": "FLASH",
}

CONVERTERS = {
    "train-iters": 100,
    "lr-decay-iters": 100,
    "eval-iters": 0,
}

CONTEXT_LENGTHS = ["32K", "64K", "128K", "256K", "512K", "1M"]
ARCHITECTURES = ["sh1", "sh2", "transformer"]
OPERATOR_CONFIG_KEY = "operator-config"
TRANFORMER_OPERATOR_CONFIG = [[['flash_te'], 50]]
ZERO_OPTIMIZER_KEY = "zero_optimization"
SUBGROUP_SIZE_KEY = "sub_group_size"
DEFAULT_SUBGROUP_SIZE = 1000000000000

def convert_single(source_config_path, context_length, architecture, output_dir, use_flow_style=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"40b-{architecture}-{context_length}.yml")
    
    with open(source_config_path, "r") as f:
        config = yaml.load(f)

    for key, value in CONVERTERS.items():
        prev_value = config.get(key, None)
        print(f"  changing {key} from {prev_value} to {value}")
        config[key] = value

    for key in KEYS_TO_DELETE:
        if key in config:
            print(f"  deleting {key}")
            del config[key]

    for key, value in KEYS_TO_INSERT.items():
        print(f"  inserting {key} with value {value}")
        config[key] = value


    # if architecture == "transformer":
    print(f"  setting flow style to {use_flow_style}")
    yaml.default_flow_style = use_flow_style
    # Change architecture-specific operator configs
    if architecture == "transformer":
        print(f"  replacing operator-config with {TRANFORMER_OPERATOR_CONFIG}")
        config["operator-config"] = TRANFORMER_OPERATOR_CONFIG
    elif architecture == "sh1":
        for i in range(len(config["operator-config"])):
            [operator], num_layers = config["operator-config"][i]
            assert num_layers == 1, "each layer must have 1 operator"
            if operator in ["hyena_se", "hyena_mr"]:
                print(f"  replacing {operator} with hyena")
                config["operator-config"][i][0][0] = "hyena"
    
    # Replace flash_v2 with flash_te
    for i in range(len(config["operator-config"])):
        [operator], num_layers = config["operator-config"][i]
        if operator == "flash_v2":
            print(f"  replacing flash_v2 with flash_te")
            config["operator-config"][i][0][0] = "flash_te"

    config[ZERO_OPTIMIZER_KEY] = { **config[ZERO_OPTIMIZER_KEY], SUBGROUP_SIZE_KEY: DEFAULT_SUBGROUP_SIZE }
    print(f"  setting {SUBGROUP_SIZE_KEY} to {DEFAULT_SUBGROUP_SIZE}")

    
    with open(output_path, "w") as f:
        yaml.dump(config, f)

    print(f"Config saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_config_dir", default="templates")
    parser.add_argument("--architecture", type=str, nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    parser.add_argument("--context_length", type=str, nargs="+", default=["32K"], choices=CONTEXT_LENGTHS)
    parser.add_argument("--block_style", action="store_true")
    parser.add_argument("--output_dir", default="generated")
    args = parser.parse_args()
    args.flow_style = not args.block_style

    for context_length in args.context_length:
        for architecture in args.architecture:
            template = os.path.join(args.source_config_dir, f"40b_{context_length}.yml")
            assert os.path.exists(template), f"Template {template} does not exist"
            print(f"Generating config for {architecture} {context_length}")
            convert_single(template, context_length, architecture, args.output_dir, use_flow_style=args.flow_style)

