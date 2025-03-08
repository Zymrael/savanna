"""
Recommended usage is not to call this script directly but to use `tools/evaluate_per_ds.sh`.

Usage: python ./launch.py \
           tools/evaluate_per_ds.py \
           -d configs \
           data/val_version_opengenome2.yml \
           model/evo2/7b_13h_8m_8s_3a_cascade15.yml

Evaluate loss per dataset.
"""
import contextlib
import io
import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir + '/..')

from savanna.arguments import GlobalConfig
from savanna.data.data_utils import build_per_dataset_val_iterators
from savanna.initialize import initialize_megatron
from savanna.training import setup_model_and_optimizer, evaluate_and_print_results, forward_step
from savanna.utils import Timers, init_wandb, get_wandb_api_key


def per_dataset_evaluation(global_config, iteration):
    """Evaluation only program, tracking each dataset loss separately.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call build_per_dataset_val_iterators to get the iterator for each val dataset.
        4) evaluate the model separately on each dataset and log the losses.
    
    Arguments:
        global_config: an instance of GlobalConfig containing the configuration
    """

    # setup logging and timers
    init_wandb(global_config=global_config)
    timers = Timers(
        use_wandb=global_config.use_wandb,
        tensorboard_writer=global_config.tensorboard_writer,
    )

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(global_config=global_config)

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        global_config=global_config,
        use_cache=False,
    )
    timers("model and optimizer").stop()

    # Data stuff.
    timers("val data iterators").start()
    val_iterators_list = build_per_dataset_val_iterators(
        global_config=global_config
    )

    timers("val data iterators").stop()

    if global_config.use_mup and global_config.coord_check:
        mup_coord_check(
            global_config,
            timers,
            lr_scheduler,
            train_data_iterator,
        )

    print(f"Evaluating iteration {iteration}...")

    prefix = "the end of training for val data. Loss with reweighting"
    for i in range(len(val_iterators_list)):
        valid_data_iterator = val_iterators_list[i]
        valid_data_iterator_name = global_config.valid_data_paths[i]
        print('Data iterator:', valid_data_iterator_name)
        evaluate_and_print_results(
            global_config=global_config,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=valid_data_iterator,
            model=model,
            iteration=iteration,
            verbose=False,
            timers=timers,
            chart_name=valid_data_iterator_name.split('/')[-3],
        )


if __name__ == "__main__":
    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()
    global_config.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    global_config.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    global_config.build_tokenizer()

    iteration = global_config.iteration

    per_dataset_evaluation(global_config=global_config, iteration=iteration)
