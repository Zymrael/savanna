"""Train"""
from savanna.neox_arguments import GlobalConfig
from savanna.training import pretrain

if __name__ == "__main__":
    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()
    global_config.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    global_config.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(global_config=global_config)
