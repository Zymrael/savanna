"""Input/output checkpointing."""

import json
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime
from glob import glob
from pathlib import Path
from pprint import pformat

import boto3
import numpy as np
import torch

from savanna import mpu, print_rank_0
from savanna.text_generation_utils import forward_model, get_batch
from savanna.utils import get_node_id, natural_sort

_checkpoint_engine = None

pool = None
delete_queue = deque()
process = None
retain_on = False


def finalize_async_save(global_config, blocking=True):
    global _checkpoint_engine, process, retain_on
    if _checkpoint_engine:
        is_finished, tag = _checkpoint_engine.commit(blocking=blocking)
        if is_finished:
            print(f"iteration: {global_config.iteration}, tag is {tag}, {is_finished}")
            if global_config.checkpoint_stores is not None and retain_on:
                process = mp.Process(
                    target=upload_checkpoint, args=(global_config, global_config.save, tag)
                )
                process.start()
                retain_on = False


def check_checkpoint_args(global_config, checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""

    assert isinstance(checkpoint_args, dict), "args stored in checkpoint is a dict"
    rank = torch.distributed.get_rank()
    for checkpoint_arg_name, checkpoint_arg_value in checkpoint_args.items():
        # ignore "max_position_embeddings", to allow context interpolation
        if checkpoint_arg_name == "max_position_embeddings":
            continue
        args_value = getattr(global_config, checkpoint_arg_name)

        error_message = (
            "LOAD_CHECKPOINT: rank{} {} value from checkpoint ({}) is not equal to the currently set argument value ({}).".format(
                rank, checkpoint_arg_name, checkpoint_arg_value, args_value
            )
        )
        if checkpoint_arg_value != args_value:
            print("LOAD_CHECKPOINT WARNING: ", error_message)
        # assert checkpoint_arg_value == args_value, error_message


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
        logits = logits.detach().cpu()[0]  # just return first batch item (they are all equal)

    return logits


def check_forward_pass(global_config, model, checkpoint_logits, inference):
    # do forward pass with loaded checkpoint
    logits = do_forward_pass(global_config=global_config, model=model, inference=inference)

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
        "mp_rank_{:02d}".format(mpu.get_model_parallel_rank() if mp_rank is None else mp_rank),
        "model_optim_rng.pt",
    )


def delete_old_checkpoints(save_dir, n_to_keep, verify_upload=False):
    if torch.distributed.get_rank() == 0:
        global pool, delete_queue
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=4)
        if len(delete_queue) > 2:
            wait(delete_queue)
            del delete_queue
            delete_queue = deque()

        ckpt_dir_regex = r"global_step[\d]*"
        if save_dir.endswith("/"):
            save_dir = save_dir.strip("/")
        all_ckpts = natural_sort(
            [i for i in glob(f"{save_dir}/*") if os.path.isdir(i) and re.search(ckpt_dir_regex, i)]
        )
        n_to_delete = len(all_ckpts) - n_to_keep
        if n_to_delete > 0:
            to_delete = all_ckpts[:n_to_delete]
            delete_queue.append(pool.submit(delete_ckpt, verify_upload, to_delete))


def delete_ckpt(verify_upload, to_delete: list):
    print(f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete)}")
    for ckpt in to_delete:
        try:
            if verify_upload and not os.path.exists(os.path.join(ckpt, '.uploaded')):
                print('Checkpoint not yet uploaded {ckpt}. Skipping')
                continue
            shutil.rmtree(ckpt)
        except FileNotFoundError:
            pass


def upload_file_s3(s3_cfg, checkpoint_dir, tag, s3_client, local_path):
    tries = 0
    while True:
        tries += 1
        try:
            s3_path = f"{s3_cfg['location']}/{tag}{local_path.replace(checkpoint_dir, '')}"
            print(f"Uploading {local_path} to { s3_cfg['bucket']}://{s3_path}", flush=True)
            s3_client.upload_file(local_path, s3_cfg["bucket"], s3_path)
            break
        except Exception as ex:
            print(ex)
            if tries > s3_cfg["max_retries"]:
                print(f"Could not upload file {local_path} after {s3_cfg['max_retries']} retries", flush=True)
                # No point in uploading partial checkpoints.
                return
            print("Error uploading file. Will retry...", flush=True)
            time.sleep(tries * 10)


def upload_checkpoint_s3(s3_cfg, checkpoint_dir, tag):
    s3_client = boto3.client(
        "s3", aws_access_key_id=s3_cfg["access_id"], aws_secret_access_key=s3_cfg["secret"]
    )
    start = datetime.now()

    print(f"Uploading {checkpoint_dir} {tag} to S3 from node {get_node_id()}")
    for local_path in Path(checkpoint_dir).rglob(f"*mp_rank_*{get_node_id()}_*/"):
        if local_path.is_file():
            local_path = str(local_path)
            upload_file_s3(s3_cfg, checkpoint_dir, tag, s3_client, local_path)

    if torch.distributed.get_rank() == 0:
        for local_path in Path(checkpoint_dir).rglob("*"):
            if local_path.is_file():
                local_path = str(local_path)
                if "_rank_" not in local_path:
                    upload_file_s3(s3_cfg, checkpoint_dir, tag, s3_client, local_path)

    end = datetime.now()
    elapsed = (end - start).total_seconds()

    print(f"Checkpoint directory {checkpoint_dir} uploaded to S3, took {elapsed} seconds", flush=True)


def upload_checkpoint_posix(store_cfg, checkpoint_dir, tag):
    if torch.distributed.get_rank() != 0:
        return

    target_path = os.path.join(store_cfg["location"], tag)
    target_path = shutil.copytree(checkpoint_dir, target_path, dirs_exist_ok=True)
    print(f"checkpoint directory {checkpoint_dir} uploaded to {target_path}")

    # Remove access checkpoints
    # delete_old_checkpoints(store_cfg.location, store_cfg.num_checkpoints)
    if store_cfg["num_checkpoints"] is not None and store_cfg["num_checkpoints"] > 0:
        chks = glob(os.path.join(store_cfg["location"], "chk_*"))
        if chks:
            chks = sorted(chks)[: -store_cfg["num_checkpoints"]]
            for chk in chks:
                print(f"Deleting old chk file {chk}")
                shutil.rmtree(chk)


def upload_checkpoint(global_config, chk_root, tag):
    """
    Upload checkpoint to a secondary location.
    """
    if tag is None:
        return

    # Only local rank 0 will upload files for all other local ranks.
    if global_config.local_rank != 0:
        return

    print(f"Uploading {tag} {global_config.iteration}  {global_config.save_retain_interval} {global_config.iteration % global_config.save_retain_interval}")

    chkpt_dir = os.path.join(chk_root, tag)
    if not os.path.exists(chkpt_dir):
        print(f"WARNING: checkpoint directory {chk_root} does not exist", flush=True)
        return

    try:
        os.makedirs(os.path.join(chk_root, ".uploading"), exist_ok=True)
        upload_indicator = Path(os.path.join(chk_root, ".uploading", f"{tag}_{get_node_id()}"))
        upload_indicator.touch()
        chkpt_dir = os.path.join(chk_root, tag)

        for store_config in global_config.checkpoint_stores:
            if store_config["storage_type"] == "s3":
                upload_checkpoint_s3(store_config, chkpt_dir, tag)
            elif store_config["storage_type"] == "posix":
                upload_checkpoint_posix(store_config, chkpt_dir, tag)
    finally:
        if upload_indicator and os.path.exists(upload_indicator):
            os.remove(upload_indicator)

        # Create a flag that delete checkpoint function can use to safely delete the folder
        if torch.distributed.get_rank() == 0:
            while True:
                if len(list(Path(os.path.join(chk_root, ".uploading")).glob(f"{tag}_*"))) == 0:
                    print(f"Uploading complete for tag {tag}", flush=True)
                    Path(os.path.join(chkpt_dir, '.uploaded')).touch()
                    break
                print(f"Waiting for {tag} to be uploaded", flush=True)
                time.sleep(10)


def save_ds_checkpoint(iteration, model, global_config):
    """Save a model checkpoint."""
    global process
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
        "data_loading": {
            "train_data_token_index": global_config.train_data_token_index,
            "train_val_test_num_samples": global_config.train_val_test_num_samples,
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

    # LoRA: merge weights into base if save_merged is True
    lora_cfg = getattr(global_config, "lora", None)
    lora_enabled = lora_cfg is not None and lora_cfg.get("enabled", False)
    save_merged = lora_enabled and lora_cfg.get("save_merged", False)
    if save_merged:
        from savanna.lora import merge_lora_weights, unmerge_lora_weights

        merge_lora_weights(model)

    # save checkpoint
    # When save_merged is True, force a blocking save to avoid a race condition:
    # async save may still be reading model memory when we unmerge weights below.
    use_async = global_config.async_save and not save_merged
    if save_merged and global_config.async_save:
        print_rank_0("LoRA: save_merged=True forces blocking save (async_save disabled for this checkpoint).")
    if use_async:
        # print(f"SAVE_CHECKPOINT: USING ASYNC SAVE")
        ret = model.save_checkpoint(global_config.save, tag=tag, client_state=sd, async_save=True)
        global _checkpoint_engine
        if _checkpoint_engine is None:
            _checkpoint_engine = ret
        if global_config.iteration % global_config.save_retain_interval == 0:
            global retain_on
            retain_on = True
    else:
        model.save_checkpoint(global_config.save, tag=tag, client_state=sd)
        if global_config.checkpoint_stores is not None:
            if global_config.iteration % global_config.save_retain_interval == 0:
                process = mp.Process(target=upload_checkpoint, args=(global_config, global_config.save, tag))
                process.start()

    # LoRA: unmerge after save if we merged, or save separate LoRA weights
    if save_merged:
        unmerge_lora_weights(model)
    elif lora_enabled:
        from savanna.lora import get_lora_state_dict

        lora_sd = get_lora_state_dict(model)
        if lora_sd:
            save_dir = os.path.join(global_config.save, tag)
            os.makedirs(save_dir, exist_ok=True)
            rank = mpu.get_model_parallel_rank()
            lora_path = os.path.join(save_dir, f"lora_mp_rank_{rank:02d}.pt")
            torch.save(lora_sd, lora_path)
            print_rank_0(f"LoRA: saved {len(lora_sd)} params to {lora_path}")

    # save config files
    if torch.distributed.get_rank() == 0 and global_config.config_files is not None:
        configs_directory = os.path.join(global_config.save, tag, "configs")
        os.makedirs(configs_directory, exist_ok=True)
        for config_filename, config_data in global_config.config_files.items():
            with open(os.path.join(configs_directory, config_filename), "w", encoding="utf-8") as f:
                if isinstance(config_data, str):
                    f.write(config_data)
                else:
                    json.dump(config_data, f)


def save_checkpoint(global_config, iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""
    start = time.time()
    if global_config.async_save:
        finalize_async_save(global_config, blocking=True)

    if global_config.deepspeed:
        save_ds_checkpoint(iteration, model, global_config)
    else:
        raise ValueError("Must be using deepspeed to use neox")

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()

    if global_config.keep_last_n_checkpoints is not None and torch.distributed.get_rank() == 0:
        verify_upload = False
        if global_config.checkpoint_stores and global_config.iteration % global_config.save_retain_interval == 0:
            verify_upload = True

        delete_old_checkpoints(global_config.save, global_config.keep_last_n_checkpoints, verify_upload=verify_upload)

    end = time.time()
    if torch.distributed.get_rank() == 0:
        print(f"SAVE_CHECKPOINT: iteration: {iteration}, ckpt saving takes {end - start} seconds")

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def _make_lora_custom_load_fn(dst_module, strict):
    """Create a custom load function that remaps base-model checkpoint keys
    to match the LoRA-wrapped model structure.

    When LoRA wraps a module ``X``, its weight moves from ``X.weight`` to
    ``X.base_layer.weight``.  A base-model checkpoint still has the old key
    names, so a naive ``load_state_dict`` with ``strict=False`` silently
    skips them, leaving the wrapped layers with random weights.

    This function builds a key mapping from the destination model and remaps
    the source state dict before loading.
    """
    from savanna.lora import LoRAColumnParallelLinear, LoRARowParallelLinear, LoRATELinear

    lora_classes = (LoRAColumnParallelLinear, LoRARowParallelLinear, LoRATELinear)

    # Build set of LoRA-wrapped module prefixes (relative to dst_module)
    lora_prefixes = set()
    for name, mod in dst_module.named_modules():
        if isinstance(mod, lora_classes):
            lora_prefixes.add(name + ".")

    def custom_load_fn(src, dst):
        if not lora_prefixes:
            dst.load_state_dict(src, strict=strict)
            return

        remapped = {}
        remap_count = 0
        for key, value in src.items():
            new_key = key
            for prefix in lora_prefixes:
                if key.startswith(prefix) and ".base_layer." not in key:
                    # Insert "base_layer." after the LoRA module prefix
                    suffix = key[len(prefix):]
                    new_key = prefix + "base_layer." + suffix
                    remap_count += 1
                    break
            remapped[new_key] = value

        if remap_count > 0:
            print_rank_0(f"LoRA checkpoint load: remapped {remap_count} keys to base_layer.*")

        # Handle size mismatches (e.g. pad_mlp_weights padding, seq_length
        # changes for Hyena filters) by manually copying with padding/truncation
        # before calling load_state_dict, which would otherwise raise.
        dst_sd = dst.state_dict()
        size_mismatch_keys = []
        for key in list(remapped.keys()):
            if key in dst_sd and isinstance(remapped[key], torch.Tensor) and remapped[key].shape != dst_sd[key].shape:
                src_tensor = remapped.pop(key)
                dst_tensor = dst_sd[key]
                # Copy overlapping region (handles both padding and truncation)
                slices = tuple(slice(0, min(s, d)) for s, d in zip(src_tensor.shape, dst_tensor.shape))
                dst_tensor.zero_()
                dst_tensor[slices].copy_(src_tensor[slices])
                size_mismatch_keys.append(key)
        if size_mismatch_keys:
            print_rank_0(
                f"LoRA checkpoint load: handled {len(size_mismatch_keys)} size-mismatched keys "
                f"via padded/truncated copy (first 5: {size_mismatch_keys[:5]})"
            )

        missing, unexpected = dst.load_state_dict(remapped, strict=False)
        # LoRA params (lora_A, lora_B) will be "missing" from a base checkpoint — that's expected
        lora_missing = [k for k in missing if "lora_" not in k and k not in size_mismatch_keys]
        if lora_missing:
            print_rank_0(f"WARNING: {len(lora_missing)} non-LoRA keys missing after remapped load: {lora_missing[:10]}")
        if unexpected:
            print_rank_0(f"WARNING: {len(unexpected)} unexpected keys in checkpoint: {unexpected[:10]}")

    return custom_load_fn


def load_checkpoint(global_config, model, optimizer, lr_scheduler, inference=False, iteration=None):
    """Load a model checkpoint and return the iteration."""
    iteration = global_config.iteration if iteration is None else iteration
    rank = torch.distributed.get_rank()

    if global_config.deepspeed:
        load_optim_and_scheduler = (
            not global_config.no_load_optim
        )  # TODO: These should be configured by separate args
        if global_config.finetune:
            load_optim_and_scheduler = False
        # check if we are loading a universal checkpoint
        if global_config.checkpoint is not None and global_config.checkpoint.get("load_universal", False):
            tag = f"global_step{iteration}_universal"
            if rank == 0:
                print(f"rank{rank} - LOAD_CHECKPOINT: Loading universal checkpoint with tag {tag}")
        elif iteration is not None:
            tag = f"global_step{iteration}"
            if rank == 0:
                print(f"rank{rank} - LOAD_CHECKPOINT: Loading checkpoint with tag {tag}")
        else:
            tag = None

        if rank == 0:
            print(f"rank{rank} - LOAD_CHECKPOINT: Loading checkpoint from {global_config.load} with tag {tag}, warmstart={global_config.warmstart}")

        # When finetuning a base checkpoint into a LoRA model, the checkpoint
        # keys won't have 'base_layer.' but the model expects them.  Use a
        # custom load function to remap keys.
        lora_cfg = getattr(global_config, "lora", None)
        lora_enabled = lora_cfg is not None and lora_cfg.get("enabled", False)
        custom_load_fn = None
        if lora_enabled and global_config.finetune:
            custom_load_fn = _make_lora_custom_load_fn(
                model.module, strict=global_config.checkpoint_strict_load
            )

        checkpoint_name, state_dict = model.load_checkpoint(
            global_config.load,
            load_optimizer_states=load_optim_and_scheduler,
            load_lr_scheduler_states=load_optim_and_scheduler,
            load_module_only=not load_optim_and_scheduler,
            tag=tag,
            load_module_strict=global_config.checkpoint_strict_load,
            custom_load_fn=custom_load_fn,
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
    if global_config.finetune or global_config.warmstart:
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
        check_checkpoint_args(global_config=global_config, checkpoint_args=checkpoint_args)
        print_rank_0(" > validated currently set args with arguments in the checkpoint ...")
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

    # # Set previous data loading values.
    # if "data_loading" in state_dict:
    #     checkpoint_data_loading = state_dict["data_loading"]
    #     if global_config.train_data_token_index is None:
    #         global_config.train_data_token_index = checkpoint_data_loading.get(
    #             "train_data_token_index"
    #         )
    #     if global_config.use_checkpoint_num_samples:
    #         global_config.train_val_test_num_samples = checkpoint_data_loading.get(
    #             "train_val_test_num_samples"
    #         )

    # rng states.
    should_set_rng = (not global_config.finetune and not global_config.no_load_rng) and (not global_config.warmstart and not global_config.no_load_rng)
    print_rank_0(f"LOAD_CHECKPOINT: should_set_rng={should_set_rng}, finetune={global_config.finetune}, warmstart={global_config.warmstart}, no_load_rng={global_config.no_load_rng}")
    if should_set_rng:
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

    # LoRA: load separate LoRA weights if present
    lora_cfg = getattr(global_config, "lora", None)
    if lora_cfg is not None and lora_cfg.get("enabled", False) and not lora_cfg.get("save_merged", False):
        rank = mpu.get_model_parallel_rank()
        # Determine the checkpoint directory for this tag
        if tag is not None:
            lora_path = os.path.join(global_config.load, tag, f"lora_mp_rank_{rank:02d}.pt")
        else:
            lora_path = None

        if lora_path is not None and os.path.exists(lora_path):
            from savanna.lora import load_lora_state_dict

            lora_sd = torch.load(lora_path, map_location="cpu", weights_only=True)
            load_lora_state_dict(model, lora_sd)
            print_rank_0(f"LoRA: loaded {len(lora_sd)} params from {lora_path}")
        else:
            print_rank_0("LoRA: no saved LoRA weights found, starting with zero-initialized B (no-op).")

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print("  successfully loaded {}".format(checkpoint_name))

    return iteration


def read_global_step(directory):
    pattern = re.compile(r"global_step(\d+)")

    latest_step = None
    latest_file_path = os.path.join(directory, "latest")

    # First, check if the "latest" file exists and parse the global_stepXXX line
    if os.path.exists(latest_file_path):
        with open(latest_file_path, "r") as latest_file:
            line = latest_file.readline().strip()  # Read the single line
            match = pattern.match(line)
            if match:
                latest_step = int(match.group(1))
                print(f"LOAD_CHECKPOINT: Read iteration {latest_step} from {latest_file_path}")
            else:
                raise ValueError(f"LOAD_CHECKPOINT: File 'latest' contains an invalid line: {line}")
    else:
        print(
            f"LOAD_CHECKPOINT: File 'latest' not found in {directory}, attempting to find latest iteration from global_step dirs"
        )

    latest_local_step = None

    # Walk through the directory and find the largest global_stepXXX subdirectory
    for subdir in os.listdir(directory):
        match = pattern.match(subdir)
        if match:
            step = int(match.group(1))
            if latest_local_step is None or step > latest_local_step:
                latest_local_step = step

    if latest_local_step is None:
        print(f"LOAD_CHECKPOINT: No global_step subdirectories found in {directory}")

        if latest_step is None:
            print(
                f"Neither latest nor local global_step subdirectories found in {directory}, defaulting to iteration 0"
            )
            return 0
    else:
        print(f"LOAD_CHECKPOINT: Latest local global step subdir: global_step{latest_local_step}")

    if latest_step is not None:
        if latest_local_step != latest_step:
            print(
                f"LOAD_CHECKPOINT: WARNING - The latest local subdirectory global_step ({latest_local_step}) "
                f"does not match the 'latest' file global_step ({latest_step}) --> Loading the older of the two. "
                f"If this results in error, ensure that `keep-last-n-checkpoints` > 1"
            )
            if latest_local_step > latest_step:
                latest_local_step = latest_step
    return latest_local_step


# def read_global_step(checkpoint_path, iteration_file="latest"):
#     iteration_path = os.path.join(checkpoint_path, iteration_file)

#     if not os.path.exists(iteration_path):
#         print(f"LOAD_CHECKPOINT: Could not find {iteration_path}, defaulting to iteration 0")
#         return 0

#     import re
#     pattern = re.compile(r"global_step(\d+)")
#     step_lines = []
#     with open(iteration_path, "r") as f:
#         for line in f:
#             match = pattern.search(line)
#             if match:
#                 step_lines.append(match.group(1))  # Capture the numeric part

#     # Ensure there is exactly one matching line
#     if len(step_lines) != 1:
#         raise ValueError(f"LOAD_CHECKPOINT: Expected exactly one 'global_step' line in {iteration_path}, found {len(step_lines)}")

#     step = int(step_lines[0])
#     print(f"LOAD_CHECKPOINT: Read iteration {step} from {iteration_path}")

#     return step
