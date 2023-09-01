#!/usr/bin/env python


import logging
import os

import deepspeed.launcher.runner


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    from savanna.neox_arguments import GlobalConfig
    from savanna.utils import get_wandb_api_key

    global_config = GlobalConfig.consume_deepy_args()
    deepspeed_main_args = global_config.get_deepspeed_main_args()

    # Extract wandb API key and inject into worker environments
    wandb_token = get_wandb_api_key(global_config=global_config)
    if wandb_token is not None:
        deepspeed.launcher.runner.EXPORT_ENVS.append("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_token
    print(deepspeed_main_args)
    deepspeed.launcher.runner.main(deepspeed_main_args)


if __name__ == "__main__":
    main()
