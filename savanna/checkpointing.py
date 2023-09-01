"""Input/output checkpointing."""

import json
import os
import re
import shutil
import random
import sys
import numpy as np

import torch
from glob import glob

from savanna import mpu
from savanna import print_rank_0
from savanna.utils import natural_sort
from savanna.text_generation_utils import get_batch, forward_model
from pathlib import Path
from pprint import pformat


def check_checkpoint_args(global_config, checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""

    assert isinstance(checkpoint_args, dict), "args stored in checkpoint is a dict"
    for checkpoint_arg_name, checkpoint_arg_value in checkpoint_args.items():
        args_value = getattr(global_config, checkpoint_arg_name)
        error_message = "{} value from checkpoint ({}) is not equal to the currently set argument value ({}).".format(
            checkpoint_arg_name, checkpoint_arg_value, args_value
        )
        assert checkpoint_arg_value == args_value, error_message


def do_forward_pass(global_config, model, inference=False):
    # set to eval mode
    model_was_in_train = model.training
    model.eval()

    # get context tokens
    # always forward full batch size
    context_tokens_tensor = (
        torch.arange(global_config.seq_length + 1)
        .repeat((global_config.train_micro_batch_size_per_gpu, 1))
        .cuda()
    )

    # forward
    if inference:
        tokens, attention_mask, position_ids = get_batch(
            global_config, context_tokens_tensor[:, : global_config.seq_length]
        )
        model_inputs = (
            tokens,
            position_ids,
            attention_mask,
            torch.Tensor(),
        )
        logits, _ = forward_model(global_config, model, model_inputs)
    elif global_config.is_pipe_parallel:
        data_iterator = iter([{"text": context_tokens_tensor}])
        _, logits = model.eval_batch(data_iter=data_iterator, return_logits=True)
    else:
        tokens, attention_mask, position_ids = get_batch(
            global_config, context_tokens_tensor[:, : global_config.seq_length]
        )
        logits = model((tokens, position_ids, attention_mask))

    # reset to train mode, if model was in training before
    if model_was_in_train:
        model.train()

    if logits is not None:
        logits = logits.detach().cpu()[
            0
        ]  # just return first batch item (they are all equal)

    return logits


def check_forward_pass(global_config, model, checkpoint_logits, inference):
    # do forward pass with loaded checkpoint
    logits = do_forward_pass(
        global_config=global_config, model=model, inference=inference
    )

    # check
    if (
        logits is not None and checkpoint_logits is not None
    ):  # this could be the case for non-final pipeline stages
        if not (logits == checkpoint_logits).all().item():
            if mpu.get_data_parallel_rank() == 0:
                print(
                    " > WARNING: validate_checkpoint_forward() forward after load of checkpoint does not yield exactly same result"
                )
            assert (
                torch.isclose(logits, checkpoint_logits).all().item()
            ), "validate_checkpoint_forward() forward after load of checkpoint does not yield a close result"


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration, release=False, mp_rank=None):
    """A unified checkpoint name."""
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    return os.path.join(
        checkpoints_path,
        directory,
        "mp_rank_{:02d}".format(
            mpu.get_model_parallel_rank() if mp_rank is None else mp_rank
        ),
        "model_optim_rng.pt",
    )


def delete_old_checkpoints(save_dir, n_to_keep):
    if torch.distributed.get_rank() == 0:
        ckpt_dir_regex = r"global_step[\d]*"
        if save_dir.endswith("/"):
            save_dir = save_dir.strip("/")
        all_ckpts = natural_sort(
            [
                i
                for i in glob(f"{save_dir}/*")
                if os.path.isdir(i) and re.search(ckpt_dir_regex, i)
            ]
        )
        n_to_delete = len(all_ckpts) - n_to_keep
        if n_to_delete > 0:
            to_delete = all_ckpts[:n_to_delete]
            print(f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete)}")
            for ckpt in to_delete:
                try:
                    shutil.rmtree(ckpt)
                except FileNotFoundError:
                    pass


def save_ds_checkpoint(iteration, model, global_config):
    """Save a model checkpoint."""
    sd = {
        "iteration": iteration,
        "args": {
            "num_layers": global_config.num_layers,
            "hidden_size": global_config.hidden_size,
            "num_attention_heads": global_config.num_attention_heads,
            "max_position_embeddings": global_config.max_position_embeddings,
            "make_vocab_size_divisible_by": global_config.make_vocab_size_divisible_by,
            "padded_vocab_size": global_config.padded_vocab_size,
            "tokenizer_type": global_config.tokenizer_type,
            "model_parallel_size": global_config.model_parallel_size,
        },
    }
    # rng states.
    if not global_config.no_save_rng:
        sd["random_rng_state"] = random.getstate()
        sd["np_rng_state"] = np.random.get_state()
        sd["torch_rng_state"] = torch.get_rng_state()
        sd["cuda_rng_state"] = torch.cuda.get_rng_state()
        sd["rng_tracker_states"] = mpu.get_cuda_rng_tracker().get_states()

    if global_config.checkpoint_validation_with_forward_pass:
        logits = do_forward_pass(global_config=global_config, model=model)
        sd["checkpoint_validation_logits"] = logits

    # checkpoint folder name
    tag = f"global_step{iteration}"

    # save checkpoint
    model.save_checkpoint(global_config.save, tag=tag, client_state=sd)

    # save config files
    if torch.distributed.get_rank() == 0 and global_config.config_files is not None:
        configs_directory = os.path.join(global_config.save, tag, "configs")
        os.makedirs(configs_directory, exist_ok=True)
        for config_filename, config_data in global_config.config_files.items():
            with open(os.path.join(configs_directory, config_filename), "w") as f:
                if isinstance(config_data, str):
                    f.write(config_data)
                else:
                    json.dump(config_data, f)


def save_checkpoint(global_config, iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""

    if global_config.deepspeed:
        save_ds_checkpoint(iteration, model, global_config)
    else:
        raise ValueError("Must be using deepspeed to use neox")

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    if global_config.keep_last_n_checkpoints is not None:
        delete_old_checkpoints(
            global_config.save, global_config.keep_last_n_checkpoints
        )

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def load_checkpoint(
    global_config, model, optimizer, lr_scheduler, inference=False, iteration=None
):
    """Load a model checkpoint and return the iteration."""
    iteration = global_config.iteration if iteration is None else iteration

    if global_config.deepspeed:
        load_optim_and_scheduler = (
            not global_config.no_load_optim
        )  # TODO: These should be configured by separate args
        if global_config.finetune:
            load_optim_and_scheduler = False
        if global_config.iteration is not None:
            tag = f"global_step{iteration}"
        else:
            tag = None
        checkpoint_name, state_dict = model.load_checkpoint(
            global_config.load,
            load_optimizer_states=load_optim_and_scheduler,
            load_lr_scheduler_states=load_optim_and_scheduler,
            tag=tag,
        )

        if checkpoint_name is None:
            # if an iteration is specified, we want to raise an error here rather than
            # continuing silently, since we are trying to load a specific checkpoint
            if iteration is not None:
                available_checkpoints = sorted(
                    [
                        int(i.name.replace("global_step", ""))
                        for i in Path(global_config.load).glob("global_step*")
                    ]
                )
                raise ValueError(
                    f"Unable to load checkpoint for iteration {iteration}. \nAvailable iterations: {pformat(available_checkpoints)}"
                )
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")

            return 0  # iteration 0, if not checkpoint loaded
    else:
        raise ValueError("Must be using deepspeed to use neox")

    # Set iteration.
    if global_config.finetune:
        iteration = 0
    else:
        iteration = state_dict.get("iteration") or state_dict.get(
            "total_iters"
        )  # total_iters backward compatible with older checkpoints
        if iteration is None:
            raise ValueError(
                f"Unable to load iteration from checkpoint {checkpoint_name} with keys {state_dict.keys()}, exiting"
            )

    # Check arguments.
    if "args" in state_dict:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(
            global_config=global_config, checkpoint_args=checkpoint_args
        )
        print_rank_0(
            " > validated currently set args with arguments in the checkpoint ..."
        )
    else:
        print_rank_0(" > could not find arguments in the checkpoint for validation...")

    # Check loaded checkpoint with forward pass
    if global_config.checkpoint_validation_with_forward_pass:
        if "checkpoint_validation_logits" in state_dict:
            check_forward_pass(
                global_config=global_config,
                model=model,
                checkpoint_logits=state_dict["checkpoint_validation_logits"],
                inference=inference,
            )
            print_rank_0(" > validated loaded checkpoint with forward pass ...")
        else:
            if mpu.get_data_parallel_rank() == 0:
                print(
                    " > WARNING: checkpoint_validation_with_forward_pass is configured but no checkpoint validation data available in checkpoint {}".format(
                        checkpoint_name
                    )
                )

    # rng states.
    if not global_config.finetune and not global_config.no_load_rng:
        try:
            random.setstate(state_dict["random_rng_state"])
            np.random.set_state(state_dict["np_rng_state"])
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            mpu.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print("  successfully loaded {}".format(checkpoint_name))

    return iteration
