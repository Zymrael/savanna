#!/usr/bin/env python
import logging
import os
import sys

from dataclasses import dataclass, fields
import argparse
import yaml
import lazy_import_plus as lazy_import

lazy_import.lazy_module("deepspeed.launcher.runner")
import deepspeed.launcher.runner

from savanna.arguments import GlobalConfig
from savanna.utils import get_wandb_api_key

@dataclass
class FwdPassArgs:
    # batch_size: int = 1
    # model_name: str = "evo2_7b_1m"
    # input_path: str = ""
    # output_path: str = "."

    @classmethod
    def from_global_config(cls, global_config: GlobalConfig):
        return cls(**{
            f.name: getattr(global_config, f.name, getattr(cls, f.name))
            for f in fields(cls)
        })
    
    @classmethod
    def add_args_to_parser(cls, parser: argparse.ArgumentParser):
        for field in fields(cls):
            parser.add_argument(
                f'--{field.name}',
                type=field.type,
                default=field.default,
                help=f'Forward pass {field.name}'
            )
        return parser

def parse_fwd_args():
    # Create parser for our custom args
    parser = argparse.ArgumentParser()
    FwdPassArgs.add_args_to_parser(parser)
    
    # Parse known args and update sys.argv
    fwd_args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    
    return fwd_args

def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    
    # Parse our custom args first
    fwd_args = parse_fwd_args()
    
    # Add to environment variables for distributed processes
    # for field in fields(FwdPassArgs):
    #     env_var = f"FWD_{field.name.upper()}"
    #     deepspeed.launcher.runner.EXPORT_ENVS.append(env_var)
    #     os.environ[env_var] = str(getattr(fwd_args, field.name))
    
    # Continue with normal DeepSpeed initialization
    import sys
    config_path = 'configs/evals/7b_chimera/1M-2gpu.yml'  # or wherever it is
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add our args to the config
    for field in fields(FwdPassArgs):
        config[f"fwd_{field.name}"] = getattr(fwd_args, field.name)

    global_config = GlobalConfig.consume_deepy_args()
    # Add to global config for distributed processes
    for field in fields(FwdPassArgs):
        setattr(global_config, f"fwd_{field.name}", getattr(fwd_args, field.name))
    
    deepspeed_main_args = global_config.get_deepspeed_main_args()

    print(deepspeed_main_args)
    deepspeed.launcher.runner.main(deepspeed_main_args)

if __name__ == "__main__":
    main()
