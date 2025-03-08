"""Train"""
from savanna.arguments import GlobalConfig
from savanna.training import pretrain

from savanna.logging import init_logger

if __name__ == "__main__":

    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()
    global_config.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    global_config.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    # setup global logging for profiler
    if global_config.should_profile:
        init_logger()

    pretrain(global_config=global_config)
