"""Pretrain utilities."""
import logging
import math
import sys
from datetime import datetime
from functools import partial

import deepspeed
import numpy as np
import torch
from torch.profiler import record_function

from savanna import mpu, print_rank_0
from savanna.checkpointing import load_checkpoint, save_checkpoint
from savanna.data.data_utils import build_train_valid_test_data_iterators
from savanna.initialize import initialize_megatron
from savanna.learning_rates import AnnealingLR
from savanna.logging import tb_wandb_log, training_log
from savanna.mfu import add_model_flop_utilization_inputs
from savanna.model import (
    BackbonePipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
    mark_norms_for_sequence_parallel_grad_sync,
)
from savanna.model.backbone import cross_entropy, dpo_loss, oadm_loss
from savanna.optimizers.sophia import SophiaG
from savanna.profiler import setup_torch_profiler
from savanna.tokenizer import CharLevelTokenizer
from savanna.utils import (
    CharCounter,
    OverflowMonitor,
    Timers,
    get_diffusion_mask,
    get_ltor_masks_and_position_ids,
    get_mlm_masks,
    get_noise_scale_logger,
    get_span_masks,
    get_total_params,
    init_wandb,
    make_upper_case,
    reduce_losses,
)

logger = logging.getLogger(__name__)


class AllocStats:
    """Compares allocation stats between steps, and Prints on rank 0 when allocations have happened."""
    def __init__(self, rank):
        self.rank = rank
        self.last_stats = None

    def step(self, iteration):
        s = torch.cuda.memory_stats()
        stats = {
            'num_alloc_retries': s['num_alloc_retries'],
            'num_device_alloc': s['num_device_alloc'],
            'num_device_free': s['num_device_free'],
        }
        if iteration >= 5 and stats != self.last_stats:
            logger.warning(f"[rank={self.rank}] [gdb] Allocation stats changed between iteration {iteration-1} and {iteration}. This isn't necessarily a problem, but generally indicates high memory pressure and can result in the job running slower: {self.last_stats} -> {stats}. If this warning becomes annoying, feel free to just disable.")
        self.last_stats = stats

def mup_weights_reinit(global_config, model):
    def has_method(o, name):
        return callable(getattr(o, name, None))

    for layer in model.modules():
        # This normally would happen in set_base_shapes if we actually were able to use the MuReadout class
        if hasattr(layer, "mup_rescale_parameters") and layer.mup_rescale_parameters:
            layer._rescale_parameters()

        if has_method(layer, "mup_reinitialize_weights"):
            layer.mup_reinitialize_weights(global_config)


def save_base_shapes(global_config, base_shapes, use_cache):
    # Instantiation of the base model fails in the init function (init_functions.py) because we haven't called set_base_shapes on it at this point, so disable it temporarily here
    global_config.use_mup = False

    base_model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    if not global_config.is_pipe_parallel:
        base_model = base_model.to_sequential()

    try:
        import mup
    except ModuleNotFoundError:
        print("Please install mup https://github.com/microsoft/mup")
        raise Exception

    base_shapes = mup.get_shapes(base_model)

    del base_model

    old_hidden_size = global_config.hidden_size
    global_config.hidden_size = global_config.hidden_size * global_config.mup_width_scale

    delta_model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    if not global_config.is_pipe_parallel:
        delta_model = delta_model.to_sequential()

    delta_shapes = mup.get_shapes(delta_model)

    # change back
    global_config.use_mup = True
    global_config.hidden_size = old_hidden_size

    save_shapes = f"{global_config.base_shapes_file}.{torch.distributed.get_rank()}"
    print(f"saving base shapes at {save_shapes}")
    mup.make_base_shapes(base_shapes, delta_shapes, savefile=save_shapes)
    print(f"base shapes saved...exiting")
    sys.exit(1)


def mup_coord_check(global_config, timers, lr_scheduler, train_data_iterator):
    from mup.coord_check import plot_coord_data

    from savanna.mup_substitute import get_coord_data

    def lazy_model(hidden_size):
        def gen():
            old_hidden_size = global_config.hidden_size
            global_config.hidden_size = hidden_size

            model, optimizer, _ = setup_model_and_optimizer(global_config=global_config, use_cache=False)

            global_config.hidden_size = old_hidden_size

            return model

        return gen

    models = {}

    # Hidden size needs to be divisible by num attention heads
    for hidden_size in (global_config.num_attention_heads * (2**p) for p in range(2, 9)):
        models[hidden_size] = lazy_model(hidden_size)

    global_config.use_mup = True
    df_up = get_coord_data(global_config, timers, lr_scheduler, models, train_data_iterator, mup=True)
    global_config.use_mup = False
    df_sp = get_coord_data(global_config, timers, lr_scheduler, models, train_data_iterator, mup=False)

    plot_coord_data(df_up, save_to=f"coord_check_up.{torch.distributed.get_rank()}.jpg")
    plot_coord_data(df_sp, save_to=f"coord_check_sp.{torch.distributed.get_rank()}.jpg")

    print_rank_0("Saved coord check plots... exiting")
    sys.exit(1)


def pretrain(global_config):
    """Main training program.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model.

    Arguments:
        global_config: an instance of GlobalConfig containing the configuration for pretrain

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
    model, optimizer, lr_scheduler = setup_model_and_optimizer(global_config=global_config, use_cache=False)
    timers("model and optimizer").stop()

    # Data stuff.
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(global_config=global_config)
    timers("train/valid/test data iterators").stop()

    if global_config.use_mup and global_config.coord_check:
        mup_coord_check(global_config, timers, lr_scheduler, train_data_iterator)


    # Calculate model flops for MFU logging
    add_model_flop_utilization_inputs(global_config=global_config)
    
    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(["model and optimizer", "train/valid/test data iterators"])
    print_rank_0("training ...")

    iteration = global_config.iteration
    if global_config.do_train and global_config.train_iters > 0:
        # edge case: save step 0 checkpoint if requested and we're starting from step 0
        if global_config.save and 0 in global_config.save_iters and iteration == 0:
            save_checkpoint(
                global_config=global_config,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        iteration = train(
            global_config=global_config,
            timers=timers,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_data_iterator=train_data_iterator,
            valid_data_iterator=valid_data_iterator,
        )

    if global_config.do_valid:
        prefix = "the end of training for val data"
        evaluate_and_print_results(
            global_config=global_config,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=valid_data_iterator,
            model=model,
            iteration=iteration,
            verbose=False,
            timers=timers,
        )

    if global_config.save and iteration != 0:
        save_checkpoint(
            global_config=global_config,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    if global_config.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        evaluate_and_print_results(
            global_config=global_config,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=test_data_iterator,
            model=model,
            iteration=iteration,
            verbose=True,
            timers=timers,
            chart_name="test",
        )


def _get_batch_dpo(global_config, tokenizer, keys, data, datatype):
    """
    A data batch for DPO is formatted in a particular way, dictated by
    `tools/prepropress_data_dpo.py`.

    This function splits the DPO data by pad tokens and prepares it for `dpo_loss`.
    Of note, the returned `labels` contain both the token ids and, in the last
    column, the reference loprobs.
    """
    datatype = torch.float32  # Note that datatype is not used, force to float32.
    data_b = mpu.broadcast_data(keys, data, datatype)
    batch = data_b["text"]

    # Check if CharLevel tokenizer. Currently only the CharLevelTokenizer supports a padding token
    if isinstance(tokenizer, CharLevelTokenizer):
        pad_token = global_config.tokenizer.pad
    else:
        pad_token = None

    device = batch.device
    batch_size, length = batch.shape

    text1s, text2s, logprob1s, logprob2s = [], [], [], []
    for i in range(batch_size):
        row = batch[i]
        pads = (row == global_config.tokenizer.pad).nonzero(as_tuple=True)[0]

        text1 = row[: pads[0]]
        text2 = row[pads[0] + 1 : pads[1]]
        logprob1 = row[pads[1] + 1].item()
        logprob2 = row[pads[2] + 1].item()

        text1s.append(text1)
        text2s.append(text2)
        logprob1s.append(logprob1)
        logprob2s.append(logprob2)

    tokens = torch.nn.utils.rnn.pad_sequence(
        text1s + text2s,
        batch_first=True,
        padding_value=global_config.tokenizer.eod,
    ).long()  # Convert float32 to long.

    # Pad up to the global sequence length.
    # Note that this is different from `dpo_data_seq_length`.
    pad_length = global_config.seq_length - tokens.shape[1]
    assert pad_length >= 0, (
        "Detected sequence in DPO dataset with length {tokens.shape[1]} when the "
        "sequence length is {global_config.seq_length}."
    )
    if pad_length > 0:
        padding = (
            torch.full(
                (tokens.shape[0], pad_length),
                global_config.tokenizer.eod,
            )
            .long()
            .to(device)
        )
        tokens = torch.concat([tokens, padding], dim=1)

    logprobs_concat = logprob1s + logprob2s
    ref_logprobs = torch.tensor(logprobs_concat).unsqueeze(1).to(device)

    # Pack token labels and logprobs into a single labels tensor.
    labels = tokens[:, 1:]
    labels = torch.concat(
        [labels, ref_logprobs],
        dim=1,
    )  # Concat produces torch.float32.
    labels = labels.contiguous()

    tokens = tokens[:, :-1].contiguous()

    global_config.eod_mask_loss = True

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=global_config.tokenizer.eod,
        pad_token=pad_token,
        eod_mask_loss=global_config.eod_mask_loss,
        pad_mask_loss=global_config.pad_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def _get_batch_mlm(global_config, tokenizer, keys, data, datatype):
    """Get batch for MLM pretraining. Support function for get_batch"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # check if CharLevel tokenizer. Currently only the CharLevelTokenizer supports a padding token
    if isinstance(tokenizer, CharLevelTokenizer):
        pad_token = global_config.tokenizer.pad
    else:
        pad_token = None

    tokens_ = data_b["text"].long()

    labels = tokens_[:, :-1].contiguous().clone()
    tokens = tokens_[:, :-1].contiguous()

    tokens, loss_mask, position_ids, attention_mask = get_mlm_masks(
        data=tokens,
        eod_token=global_config.tokenizer.eod,
        pad_token=pad_token,
        eod_mask_loss=global_config.eod_mask_loss,
        pad_mask_loss=global_config.pad_mask_loss,
        padded_vocab_size=global_config.padded_vocab_size,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def _get_batch_span_mask(global_config, tokenizer, keys, data, datatype):
    """Get batch for span mask pretraining. Support function for get_batch"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # check if CharLevel tokenizer. Currently only the CharLevelTokenizer supports a padding token
    if isinstance(tokenizer, CharLevelTokenizer):
        pad_token = global_config.tokenizer.pad
    else:
        pad_token = None

    tokens_ = data_b["text"].long()

    labels = tokens_[:, :].contiguous().clone()
    tokens = tokens_[:, :].contiguous()

    if global_config.pretraining_strategy == "SPAN_R":
        randomize_unmask = True
    else:
        randomize_unmask = False

    tokens, loss_mask, position_ids, attention_mask = get_span_masks(
        data=tokens,
        eod_token=global_config.tokenizer.eod,
        pad_token=pad_token,
        eod_mask_loss=global_config.eod_mask_loss,
        pad_mask_loss=global_config.pad_mask_loss,
        randomize_unmask=randomize_unmask,
        padded_vocab_size=global_config.padded_vocab_size,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def _get_batch_oadm(global_config, tokenizer, keys, data, datatype):
    """Get batch for OADM pretraining. Support function for get_batch"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # check if CharLevel tokenizer. Currently only the CharLevelTokenizer supports a padding token
    if isinstance(tokenizer, CharLevelTokenizer):
        pad_token = global_config.tokenizer.pad
    else:
        pad_token = None

    tokens_ = data_b["text"].long()

    labels = tokens_[:, :].contiguous().clone()
    tokens = tokens_[:, :].contiguous()

    tokens, loss_mask, position_ids, attention_mask = get_diffusion_mask(
        data=tokens,
        eod_token=global_config.tokenizer.eod,
        pad_token=pad_token,
        eod_mask_loss=global_config.eod_mask_loss,
        pad_mask_loss=global_config.pad_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def _get_batch_ar(global_config, tokenizer, keys, data, datatype):
    """Get batch for AR pretraining. Support function for get_batch"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # check if CharLevel tokenizer. Currently only the CharLevelTokenizer supports a padding token
    if isinstance(tokenizer, CharLevelTokenizer):
        pad_token = global_config.tokenizer.pad
    else:
        pad_token = None

    tokens_ = data_b["text"].long()

    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=global_config.tokenizer.eod,
        pad_token=pad_token,
        eod_mask_loss=global_config.eod_mask_loss,
        pad_mask_loss=global_config.pad_mask_loss,
        materialize_attn_mask=global_config.materialize_attn_mask,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(global_config, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    if global_config.alignment_method == "dpo":
        _get_batch = _get_batch_dpo
    elif global_config.pretraining_strategy == "AR":
        _get_batch = _get_batch_ar
    elif global_config.pretraining_strategy == "MLM":
        _get_batch = _get_batch_mlm
    elif global_config.pretraining_strategy == "SPAN" or global_config.pretraining_strategy == "SPAN_R":
        _get_batch = _get_batch_span_mask
    elif global_config.pretraining_strategy == "OADM":
        _get_batch = _get_batch_oadm

    return _get_batch(
        global_config=global_config,
        tokenizer=global_config.tokenizer,
        keys=keys,
        data=data,
        datatype=datatype,
    )


def get_batch_pipe(data, global_config, curr_scheduler=None):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    if global_config.alignment_method == "dpo":
        _get_batch = _get_batch_dpo
    elif global_config.pretraining_strategy == "AR":
        _get_batch = _get_batch_ar
    elif global_config.pretraining_strategy == "MLM":
        _get_batch = _get_batch_mlm
    elif global_config.pretraining_strategy == "SPAN" or global_config.pretraining_strategy == "SPAN_R":
        _get_batch = _get_batch_span_mask
    elif global_config.pretraining_strategy == "OADM":
        _get_batch = _get_batch_oadm

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
        global_config, global_config.tokenizer, keys, data, datatype
    )

    if curr_scheduler is not None:
        # iteration + 1 to align with how/when DeepSpeed updates the buffers
        curriculum_seqlen = curr_scheduler.update_difficulty(global_config.iteration + 1)
        if curriculum_seqlen < tokens.size()[1]:
            # seqlen-based curriculum learning
            # input_ids, position_ids, labels have size [batch size, seqlen]
            # input_ids = input_ids[:, :curriculum_seqlen].contiguous()
            tokens = tokens[:, :curriculum_seqlen].contiguous()
            position_ids = position_ids[:, :curriculum_seqlen].contiguous()
            if labels is not None:
                labels = labels[:, :curriculum_seqlen].contiguous()
            if loss_mask is not None:
                loss_mask = loss_mask[:, :curriculum_seqlen].contiguous()
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()

    # unpack data
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def forward_step(data_iterator, model, global_config, timers, return_logits=False, is_train=False):
    """Forward step."""
    if global_config.is_pipe_parallel:
        return model.eval_batch(data_iterator, return_logits=return_logits)

    # Get the batch.
    if timers is not None:
        timers("batch generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        global_config=global_config, data_iterator=data_iterator
    )

    if global_config.to_upper == "upper" or global_config.to_upper == "weighted":
        tokens, _ = make_upper_case(tokens)
        labels, lowercase_mask = make_upper_case(labels)

    if timers is not None:
        timers("batch generator").stop()

    outputs = model((tokens, position_ids, attention_mask), global_config=global_config)
    if (
        is_train
        and global_config.curriculum_learning
        and global_config.curriculum_seqlen < global_config.seq_length
    ):
        loss_mask = loss_mask[:, : global_config.curriculum_seqlen].contiguous()
        labels = labels[:, : global_config.curriculum_seqlen].contiguous()

    if global_config.alignment_method is None:

        if global_config.pretraining_strategy == "OADM":
            loss = oadm_loss(outputs, (labels, loss_mask), _fp16=global_config.fp16_lm_cross_entropy)
        else:
            if global_config.to_upper == "weighted":
                upper_loss_mask = loss_mask.bool() & (~lowercase_mask.bool())
                lower_loss_mask = loss_mask.bool() & lowercase_mask.bool()

                upper_loss_mask = upper_loss_mask.to(labels.dtype)
                lower_loss_mask = lower_loss_mask.to(labels.dtype)

                losses = cross_entropy(
                    outputs, (labels, loss_mask), _fp16=global_config.fp16_lm_cross_entropy, reduce=False
                )

                upper_loss_mask = upper_loss_mask.view(-1)
                lower_loss_mask = lower_loss_mask.view(-1)
                upper_case_loss = torch.sum(losses.view(-1) * upper_loss_mask.view(-1))
                lower_case_loss = torch.sum(losses.view(-1) * lower_loss_mask.view(-1))

                combined_loss = (
                    upper_case_loss + global_config.lowercase_loss_reweighting * lower_case_loss
                ) / (upper_loss_mask.sum() + lower_loss_mask.sum())

                return combined_loss

            else:
                loss = cross_entropy(outputs, (labels, loss_mask), _fp16=global_config.fp16_lm_cross_entropy)

    elif global_config.alignment_method == "dpo":
        loss = dpo_loss(
            outputs,
            (labels, loss_mask),
            _fp16=global_config.fp16_lm_cross_entropy,
            beta=global_config.dpo_beta,
        )
    else:
        raise ValueError(f"Invalid alignment_method {global_config.alignment_method}.")

    if return_logits:
        return loss, outputs
    return loss


def get_model(global_config, use_cache=False):
    """Build the model."""

    # Temporarily disable mup so that the base model does not use the mup init functions before set_base_shapes is called below.
    # If mup isn't being used anyways, this has no effect.
    old_use_mup = global_config.use_mup
    global_config.use_mup = False
    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    ### soft prompt tuning stuff ###
    if global_config.soft_prompt_tuning is not None and global_config.soft_prompt_tuning.get(
        "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            global_config,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=global_config.soft_prompt_tuning.get("n_tokens", 10),
            init_string=global_config.soft_prompt_tuning.get("init_string", ""),
            init_range=global_config.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not global_config.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()

    global_config.use_mup = old_use_mup

    if global_config.use_mup:
        try:
            import mup
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception

        base_shapes = f"{global_config.base_shapes_file}.{torch.distributed.get_rank()}"

        if global_config.save_base_shapes:
            save_base_shapes(global_config, base_shapes, use_cache)

        mup.set_base_shapes(model, base_shapes)

        # Call the mup replacement init functions on the model now that set_base_shapes has given each weight a .infshape attribute
        mup_weights_reinit(global_config, model)

    if global_config.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model, global_config):
    """Set up the optimizer."""
    if global_config.no_load_optim:
        return None, None
    # Build parameter groups (weight decay and non-decay).
    param_groups = get_params_for_weight_decay_optimization(model, global_config)
    print_rank_0(
        f'Configuring Optimizer type: {global_config.optimizer_type} with params: {global_config.optimizer["params"]}'
    )

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False

    # Filter out params that don't require a grad (for soft prompt tuning, etc.)
    _param_groups = []
    for param_group in param_groups:
        trainable_params = [p for p in param_group["params"] if p.requires_grad]
        param_group["params"] = trainable_params
        _param_groups.append(param_group)
    param_groups = _param_groups

    # If we're using mup, then the optimizer must be adam or sgd
    assert not global_config.use_mup or (
        global_config.optimizer_type.lower() == "adam" or global_config.optimizer_type.lower() == "sgd"
    ), f"If use_mup == True, you must specify either the adam or sgd optimizers. You passed: {global_config.optimizer_type.lower()}"

    if global_config.optimizer_type.lower() in ["cpu_adam", "cpu_torch_adam"]:
        if global_config.optimizer == "cpu_torch_adam":
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups,
            weight_decay=global_config.weight_decay,
            **global_config.optimizer["params"],
        )
    elif global_config.optimizer_type.lower() == "onebitadam":
        assert global_config.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif global_config.optimizer_type.lower() == "sm3":
        from .optimizers.sm3 import SM3

        optimizer = SM3(param_groups, **global_config.optimizer["params"])
    elif global_config.optimizer_type.lower() == "madgrad_wd":
        from .optimizers.sm3 import madgrad_wd

        optimizer = madgrad_wd(
            param_groups,
            weight_decay=global_config.weight_decay,
            **global_config.optimizer["params"],
        )
    elif global_config.optimizer_type.lower() == "adam":
        # Use Adam

        if global_config.use_mup:
            try:
                from mup import MuAdam

                adam_optimizer = MuAdam
            except ModuleNotFoundError:
                print("Please install mup https://github.com/microsoft/mup")
                raise Exception
        else:
            if global_config.use_bnb_optimizer:
                try:
                    import bitsandbytes as bnb

                    adam_optimizer = bnb.optim.Adam8bit
                except ModuleNotFoundError:
                    print(
                        "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                    )
                    raise Exception
            else:
                try:
                    # default to apex as it's slightly faster
                    from apex.optimizers import FusedAdam as Adam
                except ImportError:
                    # if apex isn't installed, use deepspeed's FusedAdam
                    print("WARNING: APEX not installed - defaulting to deepspeed's fused adam")
                    from deepspeed.ops.adam import FusedAdam as Adam
                # from deepspeed.ops.adam import FusedAdam as Adam
                # from torch.optim import Adam
                adam_optimizer = Adam
        optimizer = adam_optimizer(
            param_groups,
            weight_decay=global_config.weight_decay,
            **global_config.optimizer["params"],
        )
    elif global_config.optimizer_type.lower() == "sgd":
        try:
            from mup import MuSGD
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception
        optimizer = MuSGD(
            param_groups,
            weight_decay=global_config.weight_decay,
            **global_config.optimizer["params"],
        )
    elif global_config.optimizer_type.lower() == "sophia":
        optimizer = SophiaG(
            param_groups,
            weight_decay=global_config.weight_decay,
            batch_size=global_config.train_batch_size,
            **global_config.optimizer["params"],
        )
    else:
        raise ValueError(f"Optimizer type {global_config.optimizer_type} not recognized")

    if global_config.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer, global_config):
    """Build the learning rate scheduler."""
    if global_config.no_load_optim:
        # TODO: this should be configured as a separate arg
        return None
    if global_config.deepspeed and global_config.optimizer_type.lower() == "onebitadam":
        print_rank_0(
            "WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
            "Make sure one is added to your deepspeed config"
        )
        return None

    # Add linear learning rate scheduler.
    if global_config.lr_decay_iters is not None:
        num_iters = global_config.lr_decay_iters
    else:
        num_iters = global_config.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = global_config.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=global_config.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=global_config.lr_decay_style,
        last_iter=init_step,
        min_lr=global_config.min_lr,
        use_checkpoint_lr_scheduler=global_config.use_checkpoint_lr_scheduler,
        override_lr_scheduler=global_config.override_lr_scheduler,
        use_mup=global_config.use_mup,
    )
    if global_config.wd_free_lr is not None:
        from savanna.schedulers import MultiLR

        partial_scheduler = partial(
            AnnealingLR,
            warmup_iter=warmup_iter,
            total_iters=num_iters,
            decay_style=global_config.lr_decay_style,
            last_iter=init_step,
            min_lr=global_config.min_lr,
            use_checkpoint_lr_scheduler=global_config.use_checkpoint_lr_scheduler,
            override_lr_scheduler=global_config.override_lr_scheduler,
            use_mup=global_config.use_mup,
        )

        lambda1 = lambda opt: partial_scheduler(opt, start_lr=global_config.lr)
        lambda2 = lambda opt: partial_scheduler(opt, start_lr=global_config.wd_free_lr)

        lr_scheduler = MultiLR(optimizer, [lambda1, lambda2])

    return lr_scheduler


def setup_model_and_optimizer(global_config, use_cache=False, iteration=None):
    """Setup model and optimizer."""
    from deepspeed.utils import groups
    if (partition_size := global_config.zero_optimization.get("zero_hpz_partition_size")) is not None:
        groups._create_zero_param_parallel_group(partition_size)
    with deepspeed.zero.Init(dtype=global_config.params_dtype, enabled=global_config.zero_optimization["stage"] == 3, zero_param_parallel_group=groups._ZERO_PARAM_INTRA_PARALLEL_GROUP, sequence_data_parallel_group=mpu.get_data_parallel_group()) as init:
        model = get_model(global_config=global_config, use_cache=use_cache)
    optimizer, param_groups = get_optimizer(model=model, global_config=global_config)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, global_config=global_config)

    if global_config.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        if global_config.no_load_optim:
            assert optimizer is None
            _model_params = None
            _lr_scheduler = None
        else:
            _model_params = param_groups if optimizer is None else None
            _lr_scheduler = lr_scheduler

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=global_config,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            mpu=mpu if not global_config.is_pipe_parallel else None,
        )
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        mark_norms_for_sequence_parallel_grad_sync(model, global_config)

        if global_config.is_pipe_parallel:
            model.set_has_attention_mask(True)
            if global_config.curriculum_learning:
                from deepspeed.runtime.data_pipeline.curriculum_scheduler import (
                    CurriculumScheduler,
                )

                curr_scheduler = CurriculumScheduler(global_config.curriculum_learning)
                if iteration is not None and iteration > 0:
                    curr_scheduler.update_difficulty(iteration)
            else:
                curr_scheduler = None
            model.set_batch_fn(
                partial(
                    get_batch_pipe,
                    global_config=global_config,
                    curr_scheduler=curr_scheduler,
                )
            )
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if global_config.load is not None:
        global_config.iteration = load_checkpoint(
            global_config=global_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iteration=iteration,
        )
        print_rank_0(f"Loading checkpoint and starting from iteration {global_config.iteration}")
    else:
        global_config.iteration = 0

    # need this for correct lr scheduling resume from ckpt
    # current patch in deepspeed doesn't handle this correctly (eventually fixed in later versions)
    lr_scheduler.optimizer = model.optimizer
    lr_scheduler.param_groups = model.optimizer.param_groups
    lr_scheduler.model = model

    return model, optimizer, lr_scheduler


def backward_step(global_config, timers, optimizer, model, loss):
    """Backward step."""

    # Backward pass.
    timers("backward-backward").start()
    if global_config.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers("backward-backward").stop()

    if global_config.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


def train_step(global_config, timers, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""

    if global_config.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            global_config=global_config,
            timers=timers,
            model=model,
            data_iterator=data_iterator,
        )
    else:
        losses = []

        for _ in range(global_config.gradient_accumulation_steps):
            # Forward model for one step.
            timers("forward").start()
            with record_function("TRAIN_STEP_FORWARD"):
                if global_config.to_upper == "weighted":
                    loss = forward_step(
                        global_config=global_config,
                        timers=timers,
                        data_iterator=data_iterator,
                        model=model,
                        is_train=True,
                    )
                else:
                    loss = forward_step(
                        global_config=global_config,
                        timers=timers,
                        data_iterator=data_iterator,
                        model=model,
                        is_train=True,
                    )
            timers("forward").stop()
            losses.append(loss)

            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            with record_function("TRAIN_STEP_BACKWARD"):
                backward_step(
                    global_config=global_config,
                    timers=timers,
                    optimizer=optimizer,
                    model=model,
                    loss=loss,
                )
            timers("backward").stop()
            # Update parameters.
            timers("optimizer").start()
            with record_function("TRAIN_STEP_OPTIMIZER_STEP"):
                if global_config.deepspeed:
                    model.step()
                else:
                    raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()
        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean(),
        }  # reduces losses across machines for logging

    if global_config.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


def train_step_pipe(global_config, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine."""
    assert global_config.deepspeed

    with record_function("TRAIN_STEP_PIPE"):
        loss = model.train_batch(data_iter=data_iterator)
    # additiona_losses = model.get_additional_losses()

    loss_dict = {"lm_loss": loss}
    # loss_dict = {**loss_dict, **additiona_losses}

    # if global_config.to_upper == 'weighted':
    #     loss_dict['upper_loss'] = sub_losses[0]
    #     loss_dict['lower_loss'] = sub_losses[1]
    # Don't break Megatron's timers because we changed code paths.
    for t in [
        "forward",
        "backward",
        "allreduce",
        "optimizer",
        "batch generator",
        "data loader",
    ]:
        timers(t).reset()
    return loss_dict


def train(
    global_config,
    timers,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
):
    """Train the model function."""
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = global_config.iteration

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if global_config.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(global_config)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)

    # Initialize torch.profiler
    # if profiling is not enabled, will return a null context / no-op profiler
    profiler = setup_torch_profiler(
        enable=global_config.should_profile,
        cpu=global_config.profile_cpu,
        cuda=global_config.profile_cuda,
        profile_memory=global_config.profile_memory,
        with_stack=global_config.profile_with_stack,
        record_shapes=global_config.profile_record_shapes,
        with_flops=global_config.profile_with_flops,
        wait=global_config.profiler_schedule_wait,
        warmup=global_config.profiler_schedule_warmup,
        active=global_config.profiler_schedule_active,
        repeat=global_config.profiler_schedule_repeat,
        output_dir=global_config.profiler_output_dir,
        clean_output_dir=global_config.profiler_clean_output_dir,
        num_rows=global_config.profiler_num_rows,
    )
    alloc_stats = AllocStats(torch.distributed.get_rank())
    if isinstance(profiler, tuple):
        profiler = profiler[0]
    profiler.start()

    while iteration < global_config.train_iters:
        profiler.step()
        alloc_stats.step(iteration)

        loss_dict, skipped_iter = train_step(
            global_config=global_config,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        iteration += 1
        global_config.iteration = iteration

        if global_config.train_data_token_index is not None:
            global_config.train_data_token_index += (
                global_config.train_batch_size * global_config.seq_length
            )

        if global_config.precision == "fp16":
            overflow_monitor.check(skipped_iter)  # check for repeated overflow

        if global_config.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            global_config=global_config,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if global_config.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
        )

        # Checkpointing
        if global_config.save and iteration in global_config.save_iters:
            save_checkpoint(
                global_config=global_config,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        # Evaluation
        if (
            global_config.eval_interval
            and iteration % global_config.eval_interval == 0
            and global_config.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            evaluate_and_print_results(
                global_config=global_config,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterator=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )

        if global_config.exit_interval and iteration % global_config.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(rank, time_str, iteration)
            )
            sys.exit()

    return iteration


def evaluate(global_config, forward_step_fn, data_iterator, model, verbose=False, timers=None):
    """Evaluation.
    global_config: NeoX Arguments
    forward_step_fn: function with args `global_config, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    if global_config.char_level_ppl:
        data_iterator = CharCounter(data_iterator, global_config.tokenizer)

    with torch.no_grad():
        iteration = 0
        while iteration < global_config.eval_iters:
            iteration += 1
            if verbose and iteration % global_config.log_interval == 0:
                print_rank_0("Evaluating iter {}/{}".format(iteration, global_config.eval_iters))

            # although we're not accumulating gradients here, we count one iter as train_batch_size_per_gpu * g.a.s
            # to be consistent with deepspeed's pipe parallel engine
            # since pipe parallel already takes gas into account - default to 1 here if pipe parallel is true
            for _ in range(
                1 if global_config.is_pipe_parallel else global_config.gradient_accumulation_steps
            ):
                # Forward evaluation
                loss = forward_step_fn(
                    model=model,
                    data_iterator=data_iterator,
                    global_config=global_config,
                    timers=timers,
                )
                losses.append(loss)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if global_config.deepspeed and global_config.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {"lm_loss": reduce_losses(losses).mean().item()}
    eval_results["lm_loss_ppl"] = math.exp(eval_results["lm_loss"])

    if global_config.char_level_ppl:
        # calculate character level perplexity, if specified
        # if global_config.char_level_ppl:
        # unwrap the data_iterator
        tokens_per_char = data_iterator.tokens_per_char()
        print_rank_0(f"Counting chars took {data_iterator.total_time} seconds")

        data_iterator = data_iterator.data_iterator
        eval_results["lm_loss_char_lvl_ppl"] = math.exp(eval_results["lm_loss"] * tokens_per_char)

    # if global_config.eval_tasks:
    #     eval_results.update(
    #         run_eval_harness(
    #             model,
    #             forward_step_fn,
    #             global_config,
    #             eval_tasks=global_config.eval_tasks,
    #         ).get("results")
    #     )
    # Move model back to the train mode.
    model.train()
    return eval_results


def evaluate_and_print_results(
    global_config,
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    verbose=False,
    timers=None,
    chart_name="validation",
):
    """Helper function to evaluate and dump results on screen."""
    total_loss_dict = evaluate(
        global_config=global_config,
        forward_step_fn=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        verbose=verbose,
        timers=timers,
    )
    string = f" {chart_name} results at {prefix} | "
    for k, v in total_loss_dict.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                k3 = "_".join([k, k2])
                string += f"{k3} value: {v2:.6E} | "
                tb_wandb_log(
                    f"{chart_name}/{k3}",
                    v2,
                    iteration,
                    use_wandb=global_config.use_wandb,
                    tensorboard_writer=global_config.tensorboard_writer,
                )
        else:
            string += f"{k} value: {v:.6E} | "
            tb_wandb_log(
                f"{chart_name}/{k}",
                v,
                iteration,
                use_wandb=global_config.use_wandb,
                tensorboard_writer=global_config.tensorboard_writer,
            )

    length = len(string) + 1
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)
