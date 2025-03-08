"""Train"""
import os
import socket
from datetime import datetime

print(f"[Starting executing] datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

from savanna.arguments import GlobalConfig
from savanna.distributed import check_distributed_vars
from savanna.logging import init_logger
from savanna.training import pretrain

if __name__ == "__main__":
    
    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()

    #@jeromeku ensure env vars are set correctly when using srun_launcher with torchrun
    if global_config.use_srun_launcher:
        check_distributed_vars(assert_all=True)
        
    global_config.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    global_config.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    # setup global logging for profiler
    if global_config.should_profile:
        init_logger()

    pretrain(global_config=global_config)
