"""General utilities."""

import datetime
import math
import os
import re
import sys
import time
from functools import cache
from typing import Dict, List, Tuple

import requests

from savanna import mpu

try:
    import wandb
except ModuleNotFoundError:
    pass

from collections import deque

import deepspeed.launcher.runner as runner
import torch

from savanna import print_rank_0
from savanna.scaling.critical_batch import CriticalBatchEstimator

# Device to allocate parameters (and temporaries used to calculate the parameters)
ALLOC_DEVICE = "cuda"


def initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1, rng_fork=True):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    if rng_fork:
        with mpu.get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        init_method(weight)


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()
    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    print_rank_0(string)


def replace_nonDNA_tokens(batch, N=1):
    valid_tokens = torch.tensor([65, 84, 67, 71, 0, 48], device=batch.device, dtype=batch.dtype)

    # Create a boolean mask where valid tokens are True
    valid_mask = (batch[..., None] == valid_tokens).any(-1)

    # Replace invalid tokens with 'N'
    batch[~valid_mask] = N
    return batch


def make_upper_case(tokens):
    """
    Replace lowercase ASCII characters with uppercase.
    """
    # tokens, labels, loss_mask, attention_mask, position_ids = batch

    lowercase_mask = (tokens >= 97) & (tokens <= 122)
    uppercase_tensor = tokens.clone()
    uppercase_tensor[lowercase_mask] -= 32

    return uppercase_tensor, lowercase_mask


def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5


def mask_helper(mask, prob):
    ### adapted from lucidrains mlm_pytorch

    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def randomize_and_unmask(
    masked_tokens, data, masking_indices, random_percent, unmask_percent, specific_values
):
    """MLM style randomization and unmasking of a set percent of masked tokens"""
    device = masked_tokens.device
    randomize_mask = torch.bernoulli(torch.full(masked_tokens.shape, random_percent, device=device)).bool()
    randomize_mask = randomize_mask & masking_indices
    keep_mask = torch.bernoulli(torch.full(masked_tokens.shape, unmask_percent, device=device)).bool()
    keep_mask = keep_mask & masking_indices

    # Replace % of masked with random tokens
    indices = torch.randint(low=0, high=len(specific_values), size=masked_tokens.shape, device=device)
    random_tokens = specific_values[indices]

    masked_tokens[randomize_mask] = random_tokens[randomize_mask]

    # Revert % of masked tokens which should be unchanged
    masked_tokens[keep_mask] = data[keep_mask]

    return masked_tokens


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    pad_token,
    eod_mask_loss=False,
    pad_mask_loss=False,
    materialize_attn_mask=True,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    if materialize_attn_mask:
        # Attention mask (lower triangular).
        attention_mask = get_attn_mask(
            seq_length=seq_length,
            device=data.device,
        )
    else:
        # dummy, deepspeed expects a bool tensor returned
        attention_mask = torch.tensor([True], dtype=torch.bool, device=data.device)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    # torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0
    if pad_mask_loss:
        loss_mask[data == pad_token] = 0.0
    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids


def get_mlm_masks(
    data,
    pad_token,
    eod_token,
    eod_mask_loss,
    pad_mask_loss,
    padded_vocab_size,
    mask_percent=0.15,
    random_percent=0.1,
    unmask_percent=0.1,
):
    """
    Generate masks, position ids, and an attention mask for MLM training.

    Parameters:
        data (torch.Tensor): Input tensor containing token IDs.
        pad_token (int): Token ID used for padding.
        eod_token (int): Token ID used for end of document.
        eod_mask_loss (bool): Whether to mask the loss calculation at EOD tokens.
        pad_mask_loss (bool): Whether to mask the loss calculation at PAD tokens.
        padded_vocab_size (int): From global config, size of vocab for random tokens.
        mask_percent(float): Percent of sequence to be masked
        random_percentage (float): Percent of masked positions to be replaced with random tokens
        unmask_percentage (float): Percent of masked positions to be replaced with original

    Returns:
        tuple: (masked_tokens, loss_mask, position_ids, attention_mask)
    """

    device = data.device

    MASK_TOKEN_ID = 95  # underscore '_' is the mask token

    # 4 DNA base pairs: 65, 84, 67, 71
    # 20 Amino Acids, eos, pad: 0, 1, 65, 82, 78, 68, 67, 69, 81, 71, 72, 73, 76, 75, 77, 70, 80, 83, 84, 87, 89, 86
    # ESM2 style specific_values = torch.tensor([0, 48, 65, 82, 78, 68, 67, 69, 81, 71, 72, 73, 76, 75, 77, 70, 80, 83, 84, 87, 89, 86, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device = device)
    specific_values = torch.arange(0, padded_vocab_size, device=device)
    masked_tokens = data.clone()

    masking_indices = mask_helper(masked_tokens, mask_percent)

    if eod_mask_loss:
        masking_indices[data == eod_token] = 0  # Ignore eod
    if pad_mask_loss:
        masking_indices[data == pad_token] = 0  # Ignore pad

    # Set all masked positions to the mask token
    masked_tokens[masking_indices] = MASK_TOKEN_ID

    # Randomize and unmask a % of the masked tokens
    masked_tokens = randomize_and_unmask(
        masked_tokens, data, masking_indices, random_percent, unmask_percent, specific_values
    )

    # Generate position IDs
    position_ids = torch.arange(data.size(1), dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).repeat(data.size(0), 1)

    # Create attention_mask where True indicates the token should be attended to
    if pad_mask_loss:
        attention_mask = data != pad_token
    else:
        attention_mask = torch.ones(data.size(0), data.size(1), dtype=torch.bool, device=data.device)

    # Loss mask where True indicates the token should have a loss
    loss_mask = masking_indices
    return masked_tokens, loss_mask, position_ids, attention_mask


def get_span_masks(
    data,
    pad_token,
    eod_token,
    eod_mask_loss,
    pad_mask_loss,
    padded_vocab_size,
    mask_percent=0.15,
    p=0.2,
    max_span=10,
    randomize_unmask=False,
    random_percent=0.1,
    unmask_percent=0.1,
):
    """
    Generate masks, position ids, and an attention mask for span masking training.

    Parameters:
        data (torch.Tensor): Input tensor containing token IDs.
        pad_token (int): Token ID used for padding.
        eod_token (int): Token ID used for end of document.
        eod_mask_loss (bool): Whether to mask the loss calculation at EOD tokens.
        pad_mask_loss (bool): Whether to mask the loss calculation at PAD tokens.
        padded_vocab_size (int): From global config, size of vocab for random tokens.
        mask_percent(float): Percent of sequence to be masked
        p (float): Probability for the geometric distribution to sample span lengths.
        max_span (int): Maximum span length to be masked.

    Returns:
        tuple: (masked_tokens, loss_mask, position_ids, attention_mask)
    """

    device = data.device
    MASK_TOKEN_ID = 95  # underscore '_' is the mask token

    data = replace_nonDNA_tokens(data)

    length = data.size(1)
    batch_size = data.size(0)

    num_tokens_to_mask = int(mask_percent * length)

    masked_tokens = data.clone()
    loss_mask = torch.zeros(data.shape, dtype=torch.bool, device=device)
    attention_mask = torch.ones(data.shape, dtype=torch.bool, device=device)

    for i in range(batch_size):
        num_masked = 0
        while num_masked < num_tokens_to_mask:
            span_length = min(torch.distributions.Geometric(torch.tensor([p])).sample().item(), max_span)
            span_length = int(span_length)

            if num_masked + span_length > num_tokens_to_mask:
                span_length = num_tokens_to_mask - num_masked

            start_idx = torch.randint(0, length - span_length + 1, (1,)).item()

            masked_tokens[i, start_idx : start_idx + span_length] = MASK_TOKEN_ID
            loss_mask[i, start_idx : start_idx + span_length] = 1
            num_masked += span_length

        if pad_mask_loss:
            loss_mask[i][data[i] == pad_token] = 0
            attention_mask[i][data[i] == pad_token] = 0

        if eod_mask_loss:
            loss_mask[i][data[i] == eod_token] = 0

    if randomize_unmask:
        # Randomize and unmask a % of the masked tokens
        specific_values = torch.arange(0, padded_vocab_size, device=device)
        mask_indices = masked_tokens == MASK_TOKEN_ID

        masked_tokens = randomize_and_unmask(
            masked_tokens, data, masking_indices, random_percent, unmask_percent, specific_values
        )

    # Generate position IDs
    position_ids = torch.arange(length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

    return masked_tokens, loss_mask, position_ids, attention_mask


def get_diffusion_mask(data, pad_token, eod_token, eod_mask_loss, pad_mask_loss):
    """
    Generate masks for order-agnostic autoregressive diffusion from Hoogeboom et al. 2021.

    Parameters:
        data (torch.Tensor): Input tensor containing token IDs.
        pad_token (int): Token ID used for padding.
        eod_token (int): Token ID used for end of document.
        eod_mask_loss (bool): Whether to mask the loss calculation at EOD tokens.
        pad_mask_loss (bool): Whether to mask the loss calculation at PAD tokens.

    Returns:
        tuple: (masked_tokens, loss_mask, position_ids, attention_mask)
    """
    device = data.device

    MASK_TOKEN_ID = 95  # underscore '_' is the mask token

    length = data.size(1)
    batch_size = data.size(0)

    masked_tokens = data

    times = torch.randint(0, length, (batch_size,), dtype=torch.int64, device=device)
    num_masks = length - times + 1

    # 2d tensor of masks, random permutations, and then scatter back into the permuted positions
    masking_indices = torch.zeros((batch_size, length), dtype=torch.bool, device=device)
    perm = torch.argsort(torch.rand((batch_size, length), device=device), dim=1)
    mask_range = torch.arange(length, device=device).expand(batch_size, length)
    mask = mask_range < num_masks.unsqueeze(1)
    masking_indices.scatter_(1, perm, mask)

    if eod_mask_loss:
        masking_indices[masked_tokens == eod_token] = 0  # Ignore eod
    if pad_mask_loss:
        masking_indices[masked_tokens == pad_token] = 0  # Ignore pad

    masked_tokens[masking_indices] = MASK_TOKEN_ID

    loss_mask = masking_indices

    if not pad_mask_loss:
        attention_mask = data != pad_token
    else:
        attention_mask = torch.ones(data.size(0), data.size(1), dtype=torch.bool, device=device)

    position_ids = torch.arange(data.size(1), dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).repeat(data.size(0), 1)

    return masked_tokens, loss_mask, position_ids, attention_mask


def mask_control_tags(labels, eod_token_id=10):
    labels, loss_mask = labels[0], labels[1]

    control_tags = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    control_mask = torch.isin(labels, torch.tensor(control_tags, device=labels.device))
    loss_mask[control_mask] = 0

    tag_start_char = 124  # start and end delim: '|'
    tag_chars = {95, 59, 32}  # chars only found in control tags: _, ;, space

    phylotag_mask = mask_phylogenetic_tags(labels, tag_start_char, tag_chars, eod_token_id)
    loss_mask = loss_mask * phylotag_mask

    return loss_mask


def mask_phylogenetic_tags(tokenized_sequence, terminal_tag_char, other_tag_chars, eod_token_id):
    """
    Optimized version to create a phylonetic tag mask for batched tokenized sequences with correct handling of partial tags.
    Args:
    - tokenized_sequence (torch.Tensor): A batched tensor of shape (batch_size, seq_length).
    - terminal_tag_char (int): The token ID representing the start and end of a phylogenetic tag ('|').
    - other_tag_chars (set of int): A set of token IDs that are uniquely part of the tag ('_', ';', etc.).
    - eod_token_id (int): The token ID representing the end-of-document (EOD).
    Returns:
    - mask_vector (torch.Tensor): A batched mask of the same shape as tokenized_sequence where
      1 represents non-tag tokens and 0 represents tokens within the masked region.
    """
    device = tokenized_sequence.device
    batch_size, seq_len = tokenized_sequence.shape
    mask_vector = torch.ones_like(tokenized_sequence, dtype=torch.int, device=device)

    # To address when unbalanced tags are present
    terms = torch.tensor([0, seq_len - 1], device=device)
    other_tags = torch.tensor(list(other_tag_chars), device=device)
    for batch_idx in range(batch_size):
        tag_term_locs = torch.where(tokenized_sequence[batch_idx] == terminal_tag_char)[0]
        tag_end_locs = torch.where(tokenized_sequence[batch_idx] == eod_token_id)[0]

        merged_tags = torch.cat((terms, tag_term_locs, tag_end_locs)).sort()[0]
        merged_tags = merged_tags.unique()

        start = 0  # First and last locations are always added
        for end in merged_tags[1:]:
            if torch.isin(tokenized_sequence[batch_idx][start:end], other_tags).sum() > 0:
                # end token is not part of the tag
                if eod_token_id == tokenized_sequence[batch_idx][end]:
                    end = end - 1
                if eod_token_id == tokenized_sequence[batch_idx][start]:
                    start = start + 1

                mask_vector[batch_idx][start : (end + 1)] = 0
            start = end
    return mask_vector


@cache
def rank():
    rank = os.environ.get("RANK")
    if rank is not None:
        rank = int(rank)
    return rank


@cache
def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
    return int(local_rank)


@cache
def run_id():
    # TODO: create this in the launcher process, and inherit in launched processes
    key = "SAVANNA_RUN_ID"
    if key not in os.environ:
        # generate a new run id based on datetime
        os.environ[key] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return os.environ[key]


def is_bnb_available():
    """True if bitsandbytes optimizers are available"""
    return importlib.util.find_spec("bitsandbytes") is not None


def is_main():
    """True if is rank 0"""
    r = rank()
    assert r is not None
    return r == 0


def is_local_main():
    """True if is the local main process"""
    return local_rank() == 0


def is_mp_rank_0():
    """True if mp rank == 0"""
    return mpu.get_model_parallel_rank() == 0


def get_wandb_api_key(global_config):
    """Get Weights and Biases API key from ENV or .netrc file. Otherwise return None"""
    if "WANDB_LOCAL" in os.environ:
        return "LOCAL"
    if "WANDB_API_KEY" in os.environ:
        return os.environ["WANDB_API_KEY"]

    wandb_token = requests.utils.get_netrc_auth(global_config.wandb_host)

    if wandb_token is not None:
        return wandb_token[1]


def init_wandb(global_config):
    # Wandb. (one worker per machine)
    if global_config.use_wandb == False:
        return

    if not global_config.wandb_init_all_ranks:
        use_wandb = is_main() and (get_wandb_api_key(global_config=global_config) is not None)
        global_config.update_value("use_wandb", use_wandb)
    if global_config.use_wandb:
        group_name = global_config.wandb_group
        if global_config.wandb_run_name is None:
            name = run_id() if group_name else None
        else:
            name = global_config.wandb_run_name

        try:
            wandb.init(
                id=run_id(),
                project=global_config.wandb_project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=global_config.wandb_team,
                dir=global_config.log_dir,
            )
        except wandb.UsageError as e:
            global_config.update_value("use_wandb", False)
            print(e)
            print(
                "Skipping wandb. Execute `wandb login` on local or main node machine to enable.",
                flush=True,
            )
        wandb.config.update(global_config.all_config, allow_val_change=True)


def obtain_resource_pool(hostfile_path, include_arg, exclude_arg) -> Dict[str, List[int]]:
    """
    Get dict of `resource_pool[hostname] = [list of GPU ranks]` using hostfile, include and exclude args.
    Modified from: `deepspeed.launcher.runner.main`
    """
    resource_pool = runner.fetch_hostfile(hostfile_path)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool["localhost"] = device_count

    active_resources = runner.parse_inclusion_exclusion(resource_pool, include_arg, exclude_arg)
    return active_resources


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def ddb(rank=0):
    """
    Distributed Debugger that will insert a py debugger on rank `rank` and
    pause all other distributed processes until debugging is complete.
    :param rank:
    """
    if torch.distributed.get_rank() == rank:
        from pdb import Pdb

        pdb = Pdb(skip=["torch.distributed.*"])
        pdb.set_trace(sys._getframe().f_back)
    torch.distributed.barrier()


class Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # pollutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def expand_operator_types(operator_config, num_layers):
    """
    Expands an `operator_config` list in the following format:

        [
        [['operator_type_1', ..., `operator_type_n`], 12]
        ]

    to a flattened list of length `num_layers`.

    :param params_list:
    :return:
    """
    # if only strings are found in the config, we assume it's already expanded
    if all([isinstance(i, str) for i in operator_config]):
        return operator_config
    newlist = []
    for item in operator_config:
        # instead of specifying a number - we can specify 'all' to extend this pattern across all layers
        if item[1] == "all":
            assert num_layers % len(item[0]) == 0, (
                f"Number of layers ({num_layers}) is not divisible by the length " f"of pattern: {item[0]}"
            )
            return item[0] * (num_layers // len(item[0]))
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist


class OverflowMonitor:
    """
    Checks if the past n iterations have been skipped due to overflow, and exits
    training if that happens.
    """

    def __init__(self, optimizer, n=50):
        self.optimizer = optimizer
        self.n = n
        self.history = deque(maxlen=n)

    def check(self, skipped):
        self.history.append(skipped)
        if self.optimizer.overflow and len(self.history) == self.n and all(self.history):
            raise Exception(f"Skipped {self.n} iterations in a row due to Overflow - Exiting training.")


def get_noise_scale_logger(global_config, model):
    if global_config.log_gradient_noise_scale:
        if global_config.zero_stage > 1:
            raise NotImplementedError(
                "Gradient Noise Scale logging is currently not compatible with ZeRO stage 2+."
            )
        noise_scale_logger = CriticalBatchEstimator(
            model=model,
            batch_size_small=global_config.train_batch_size,
            n_batches=global_config.gradient_noise_scale_n_batches,
            cpu_offload=global_config.gradient_noise_scale_cpu_offload,
            global_config=global_config,
            mpu=mpu,
        )
    else:
        noise_scale_logger = None
    return noise_scale_logger


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(), params
            ),
            flush=True,
        )
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def setup_for_inference_or_eval(
    use_cache=True,
    overwrite_values=None,
):
    """
    Initializes the model for evaluation or inference (doesn't load optimizer states, etc.) from command line args.

    use_cache: bool
        Whether to use key value caching in inference.
    overwrite_values: dict
        Optional Values to overwrite in the model config.
    """

    from savanna.arguments import GlobalConfig
    from savanna.initialize import initialize_megatron
    from savanna.training import setup_model_and_optimizer

    _overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        "zero_optimization": None,  # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
    }
    if overwrite_values:
        _overwrite_values.update(overwrite_values)
    global_config = GlobalConfig.consume_global_config(overwrite_values=_overwrite_values)
    global_config.configure_distributed_args()
    global_config.build_tokenizer()

    if global_config.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize wandb
    init_wandb(global_config=global_config)

    # initialize megatron
    initialize_megatron(global_config)

    # set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(
        global_config=global_config,
        use_cache=use_cache,
        iteration=global_config.iteration,
    )  # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0("Finished loading model")

    model.module.inference_mode(use_cache=use_cache)
    return model, global_config


class CharCounter:
    """
    Wraps the data_iterator to count the number of characters in a batch
    """

    def __init__(self, data_iterator, tokenizer):
        self.tokenizer = tokenizer
        self.data_iterator = data_iterator
        self.char_count = 0
        self.batch_count = 0
        self.token_count = 0
        self.total_time = 0

    def tokens_per_char(self):
        return self.token_count / self.char_count

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        batch = self.data_iterator.__next__()
        for b in batch["text"]:
            self.token_count += len(b)
            self.char_count += len(self.tokenizer.detokenize(b.tolist()))
        self.batch_count += 1
        end = time.time()
        self.total_time += end - start
        return batch


def bp():
    import rpdb

    port = 4444
    while True:
        try:
            rpdb.set_trace(port=port)
        except OSError:
            port += 1
        else:
            break


def get_streams():
    import gc

    out = {torch.cuda.default_stream()}
    for x in gc.get_objects():
        try:
            if isinstance(x, torch.cuda.Stream):
                out.add(x)
        except:
            pass
    return out


# nvidia-resiliency-ext
def round_float_values(d: Dict) -> Dict:
    return {k: round(v, 2) for k, v in d.items()}


def print_gpu_scores(report, rank):
    print(f"=== GPUs perf scores. Report from rank {rank} ===")
    rel_scores = round_float_values(report.gpu_relative_perf_scores)
    print("GPU relative perf scores:", rel_scores)
    indiv_scores = round_float_values(report.gpu_individual_perf_scores)
    print("GPU individual perf scores:", indiv_scores)


def print_section_scores(report, rank):
    print(f"=== Sections perf scores. Report from rank {rank} ===")
    rel_scores = {}
    for section in report.section_relative_perf_scores:
        rel_scores[section] = round_float_values(report.section_relative_perf_scores[section])
    print("Sections relative perf scores:", rel_scores)
    indiv_scores = {}
    for section in report.section_individual_perf_scores:
        indiv_scores[section] = round_float_values(report.section_individual_perf_scores[section])
    print("Sections individual perf scores:", indiv_scores)


def print_stragglers(stragglers):
    # Print stragglers in easy to parse format
    for s in stragglers["straggler_gpus_relative"]:
        print(f"DETECTED RELATIVE STRAGGLER GPU RANK={s.rank} NODE={s.node}")
    for s in stragglers["straggler_gpus_individual"]:
        print(f"DETECTED INDIVIDUAL STRAGGLER GPU RANK={s.rank} NODE={s.node}")
    for section in stragglers["straggler_sections_relative"]:
        for s in stragglers["straggler_sections_relative"][section]:
            print(f"DETECTED RELATIVE STRAGGLER SECTION={section} RANK={s.rank} NODE={s.node}")
    for section in stragglers["straggler_sections_individual"]:
        for s in stragglers["straggler_sections_individual"][section]:
            print(f"DETECTED INDIVIDUAL STRAGGLER SECTION={section} RANK={s.rank} NODE={s.node}")

def print_local_summaries(report, rank):
    if report is not None:
        print(f"rank{rank}: Section Summaries")
        for k, v in report.local_section_summaries.items():
            print(f"{k}:\n\t{v}")
        print(f"rank{rank}: Kernel Summaries")
        for k, v in report.local_kernel_summaries.items():
            print(f"{k}:\n\t{v}")

def print_straggler_report(
    report,
    rank,
    iteration,
    gpu_rel_threshold=0.7,
    gpu_individual_threshold=0.7,
    section_rel_threshold=0.7,
    section_individual_threshold=0.7,
    gpu_scores=True,
    section_scores=True,
    stragglers=True
):
    if report is not None:
        print(f"STRAGGLER REPORT @ {iteration}")
        if gpu_scores:
            print_gpu_scores(report, rank)
        if section_scores:
            print_section_scores(report, rank)
        if stragglers:
            stragglers = report.identify_stragglers(
                gpu_rel_threshold=gpu_rel_threshold,
                section_rel_threshold=section_rel_threshold,
                gpu_indiv_threshold=gpu_individual_threshold,
                section_indiv_threshold=section_individual_threshold,
            )
            print("Straggler report:")
            print(stragglers)
            print_stragglers(stragglers)

def get_node_id():
    if not torch.distributed.is_initialized():
        return 0

    rank = torch.distributed.get_rank()
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    node_id = rank // local_world_size
    return node_id

# Needed for Heimdall straggler detection
class NullDetector:
    def __init__(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

# TE shape requirements
FP8_SHAPE = (8, 16)

def pad_to_multiple(d: int, multiple: int = 16) -> int:
    remainder = d % multiple
    if remainder == 0:
        return d
    padding = multiple - remainder
    return d + padding
