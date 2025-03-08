# Copyright (c) 2021, EleutherAI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import deepspeed.comm as comms
import torch

try:
    import wandb
except ModuleNotFoundError:
    pass

import logging
import os

import torch.distributed as dist

from savanna import print_rank_0
from savanna.memory_stats import gather_mem_alloc_stats, get_memory_stats
from savanna.utils import report_memory

logger = logging.getLogger()


# from torchtune.logging
def get_world_size_and_rank():
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current process in the default process group.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if dist.is_available() and dist.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0


def init_logger():
    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s | %(levelname)s | %(pathname)s:%(lineno)d: %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def log_norm(norm, key, iteration_no):
    if norm is not None:
        tb_wandb_log(
            key,
            norm,
            iteration_no,
            use_wandb=True,
            tensorboard_writer=None,
            all_ranks=False,
        )


class Tee:
    """Duplicate output to both stdout/err and file"""

    def __init__(self, file, err=False):
        self.file = open(file, "w")
        self.err = err
        if not err:
            self.std = sys.stdout
            sys.stdout = self
        else:
            self.std = sys.stderr
            sys.stderr = self

    def __del__(self):
        if not self.err:
            sys.stdout = self.std
        else:
            sys.stderr = self.std
        self.file.close()

    def write(self, data):
        try:
            self.file.write(data)
        except OSError:
            pass
        try:
            self.std.write(data)
        except OSError:
            pass

    def flush(self):
        try:
            self.file.flush()
        except OSError:
            pass


def human_readable_flops(num):
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


def get_total_flops(global_config, model):
    total_params = sum(
        p.ds_numel if getattr(p, "ds_tensor", None) is not None else p.numel() for p in model.parameters()
    )
    ff = total_params * 6
    attn = global_config.seq_length * global_config.hidden_size * global_config.num_layers * 60
    total_flops = (
        global_config.train_batch_size * global_config.seq_length * (ff + attn)
    )
    return total_flops


def get_flops(global_config, model, iter_time_s):
    world_size = torch.distributed.get_world_size()
    total_flops = get_total_flops(global_config, model)
    flops = total_flops / (iter_time_s * world_size)
    return flops


def training_log(
    global_config,
    timers,
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    model,
    optimizer,
    noise_scale_logger,
):
    """Log training information such as losses, timing, etc."""

    # Update losses.
    skipped_iters_key = "skipped iterations"
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    got_nan_key = "got nan"

    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, 0.0) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan

    total_loss_dict[got_nan_key] = total_loss_dict.get(got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    if not global_config.is_pipe_parallel:
        add_to_logging("forward")
        add_to_logging("backward")
        add_to_logging("backward-backward")
        add_to_logging("backward-allreduce")
        add_to_logging("backward-master-grad")
        add_to_logging("backward-clip-grad")
        add_to_logging("optimizer")
        add_to_logging("batch generator")

        # Log timer info to tensorboard and wandb
        normalizer = iteration % global_config.log_interval
        if normalizer == 0:
            normalizer = global_config.log_interval
        if torch.distributed.get_rank() == 0:
            timers.write(names=timers_to_log, iteration=iteration, normalizer=normalizer)
    else:
        # with pipeline parallel, the megatron timers are overridden by the deepspeed ones.
        # Try to grab timer values from model engine. Only recently added to deeperspeed, so check that the engine
        # has that attribute first
        if hasattr(model, "timer_values") and model.timer_values is not None:
            if model.wall_clock_breakdown() and model.global_steps % model.steps_per_print() == 0:
                timer_values = model.timer_values
                # deepspeed already logs to tensorboard / prints values, so just log to wandb
                if global_config.use_wandb and torch.distributed.get_rank() == 0:
                    for key in timer_values:
                        tb_wandb_log(
                            f"timers/{key}",
                            timer_values[key],
                            iteration,
                            use_wandb=global_config.use_wandb,
                            tensorboard_writer=global_config.tensorboard_writer,
                        )

    # write losses, lr, etc. every step
    tb_wandb_log(
        "train/learning_rate",
        learning_rate,
        iteration,
        use_wandb=global_config.use_wandb,
        tensorboard_writer=global_config.tensorboard_writer,
    )
    for key in loss_dict:
        tb_wandb_log(
            f'train/{key.replace(" ", "_")}',
            loss_dict[key],
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )
    if global_config.fp16:
        tb_wandb_log(
            "train/loss_scale",
            loss_scale,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )

    # log gradient noise scale
    if global_config.log_gradient_noise_scale:
        if noise_scale_logger.noise_scale is not None:
            tb_wandb_log(
                "train/noise_scale",
                noise_scale_logger.noise_scale,
                iteration,
                use_wandb=global_config.use_wandb,
                tensorboard_writer=global_config.tensorboard_writer,
            )

    # (optional) Log optimizer states to wandb / tb every step
    if global_config.log_optimizer_states:
        for k, v in optimizer.state_dict()["optimizer_state_dict"]["state"].items():
            for ki, vi in v.items():  # step, module
                if ki != "step":
                    opt_state_norm = torch.norm(vi) if hasattr(vi, "dim") else vi
                    tb_wandb_log(
                        f"optimizer_state_norms/{k}_{ki}",
                        opt_state_norm,
                        iteration,
                        use_wandb=global_config.use_wandb,
                        tensorboard_writer=global_config.tensorboard_writer,
                    )

    # (optional) Log grad/param norms to wandb / tb every step
    if global_config.log_grad_pct_zeros or global_config.log_grad_norm or global_config.log_param_norm:
        if global_config.log_grad_pct_zeros or global_config.log_grad_norm:
            model.store_gradients = True  # start storing gradients

        for i, (name, param) in enumerate(model.module.named_parameters()):
            if global_config.log_grad_pct_zeros:
                if hasattr(model, "stored_gradients") and model.stored_gradients is not None:
                    grad = model.stored_gradients[i]
                    if grad is not None:
                        tb_wandb_log(
                            f"pct_grad_zeros/{name}",
                            (grad == 0).float().mean().item() * 100,
                            iteration,
                            use_wandb=global_config.use_wandb,
                            tensorboard_writer=global_config.tensorboard_writer,
                            all_ranks=True,
                        )
            if global_config.log_grad_norm:
                if hasattr(model, "stored_gradients") and model.stored_gradients is not None:
                    grad = model.stored_gradients[i]
                    if grad is not None:
                        tb_wandb_log(
                            f"gradient_norms/{name}",
                            torch.norm(grad),
                            iteration,
                            use_wandb=global_config.use_wandb,
                            tensorboard_writer=global_config.tensorboard_writer,
                            all_ranks=True,
                        )
            if global_config.log_param_norm:
                tb_wandb_log(
                    f"parameter_norms/{name}",
                    torch.norm(param),
                    iteration,
                    use_wandb=global_config.use_wandb,
                    tensorboard_writer=global_config.tensorboard_writer,
                    all_ranks=True,
                )

    if iteration % global_config.log_interval == 0:
        # log other stuff every global_config.log_interval iters
        elapsed_time = timers("interval time").elapsed()
        iteration_time = elapsed_time / global_config.log_interval
        samples_per_sec = global_config.train_batch_size / iteration_time
        log_string = " samples/sec: {:.3f} |".format(samples_per_sec)
        tb_wandb_log(
            "runtime/samples_per_sec",
            samples_per_sec,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )
        tb_wandb_log(
            "runtime/iteration_time",
            iteration_time,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )
        log_string += " iteration {:8d}/{:8d} |".format(iteration, global_config.train_iters)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time * 1000.0 / global_config.log_interval
        )
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        num_iterations = max(1, global_config.log_interval - total_loss_dict[skipped_iters_key])

        # log curriculum learning
        if global_config.curriculum_learning:
            tb_wandb_log(
                "curriculum_seqlen",
                global_config.curriculum_seqlen,
                iteration,
                use_wandb=global_config.use_wandb,
                tensorboard_writer=global_config.tensorboard_writer,
            )

        # log tflop / gpu
        flops_per_s_per_gpu = get_flops(global_config=global_config, model=model, iter_time_s=iteration_time)
        log_string += f" approx flops per GPU: {human_readable_flops(flops_per_s_per_gpu)} |"
        tb_wandb_log(
            "runtime/flops_per_sec_per_gpu",
            flops_per_s_per_gpu,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )

        batch_size = global_config.train_batch_size * global_config.seq_length
        log_string += f" batch size (tokens): {(batch_size/1e6):.1f}M ({(batch_size*2048/torch.distributed.get_world_size()/1e6):.1f}M @ 2048 GPUs) |"
        tb_wandb_log(
            "data/global_batch_size_tokens",
            batch_size,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )

        tokens_per_second_per_gpu = batch_size / torch.distributed.get_world_size() / iteration_time
        log_string += f" tokens/s/gpu: {tokens_per_second_per_gpu:0.1f} |"
        tb_wandb_log(
            "data/tokens_per_second_per_gpu",
            tokens_per_second_per_gpu,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )

        seconds_per_month = 30 * 24 * 60 * 60
        num_gpus = 2048
        total_tokens_per_month = tokens_per_second_per_gpu * seconds_per_month * num_gpus
        total_tokens_per_month /= 1e12
        log_string += f" total_token/month @ {num_gpus} GPUs: {total_tokens_per_month:0.1f}T |"
        tb_wandb_log(
            f"data/total_tokens_per_month@{num_gpus}GPUs",
            total_tokens_per_month,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )

        if global_config.train_data_token_index is not None:
            log_string += f" token index into training data: {global_config.train_data_token_index} |"
            tb_wandb_log(
                "data/train_data_token_index",
                global_config.train_data_token_index,
                iteration,
                use_wandb=global_config.use_wandb,
                tensorboard_writer=global_config.tensorboard_writer,
            )
        # hw_model_flops_per_iteration -> model flops per iteration taking into account activation checkpointing
        # TODO: change iteration time to sum of timers/forward + backward + optimizer?
        tb_wandb_log(
            "efficiency/model_flops_per_iteration",
            global_config.model_flops_per_iteration,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )
        tb_wandb_log(
            "efficiency/hw_model_flops_per_iteration",
            global_config.hw_model_flops_per_iteration,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )

        global_flops_throughput = global_config.hw_model_flops_per_iteration / iteration_time
        device_flops_throughput = global_flops_throughput / torch.distributed.get_world_size()
        log_string += f" global flops throughput (flops / s): {global_flops_throughput:0.2e} |"
        tb_wandb_log(
            "efficiency/global_flops_throughput",
            global_flops_throughput,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )
        log_string += f" device flops throughput (flops / s): {device_flops_throughput:0.2e} |"
        tb_wandb_log(
            "efficiency/device_flops_throughput",
            device_flops_throughput,
            iteration,
            use_wandb=global_config.use_wandb,
            tensorboard_writer=global_config.tensorboard_writer,
        )
        # MFU = model_flops_throughput / theoretical_throughput where model_flops_throughput = (flops per fwd pass + flops_ per bwd pass) / iteration_time
        # Should always be less than or equal to HFU, which takes into account activation checkpointing (and other actual computations)
        model_flops_throughput = global_config.model_flops_per_iteration / (
            torch.distributed.get_world_size() * iteration_time
        )
        if global_config.theoretical_device_throughput is not None:
            theoretical_throughput = global_config.theoretical_device_throughput
            mfu = model_flops_throughput / theoretical_throughput
            log_string += f" MFU: {mfu * 100:0.2f}% |"
            tb_wandb_log(
                "efficiency/mfu",
                mfu,
                iteration,
                use_wandb=global_config.use_wandb,
                tensorboard_writer=global_config.tensorboard_writer,
            )
            hfu = device_flops_throughput / theoretical_throughput
            log_string += f" HFU: {hfu * 100:0.2f}% |"
            tb_wandb_log(
                "efficiency/hfu",
                hfu,
                iteration,
                use_wandb=global_config.use_wandb,
                tensorboard_writer=global_config.tensorboard_writer,
            )

        # Add memory stats
        # mem_stats is a dict with keys 'memory/key/sub_key' already flattened
        if global_config.log_memory_stats:
            mem_stats = get_memory_stats(include_counts=True)
            for k, v in mem_stats.items():
                tb_wandb_log(
                    k,
                    v,
                    iteration,
                    use_wandb=global_config.use_wandb,
                    tensorboard_writer=global_config.tensorboard_writer,
                )
        if global_config.log_memory_alloc_counts:
            alloc_stats = gather_mem_alloc_stats()
            if torch.distributed.get_rank() == 0:
                for k, v in alloc_stats.items():
                    tb_wandb_log(
                        k,
                        v,
                        iteration,
                        use_wandb=global_config.use_wandb,
                        tensorboard_writer=global_config.tensorboard_writer,
                    )

        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                v = (
                    total_loss_dict[key].item()
                    if hasattr(total_loss_dict[key], "item")
                    else total_loss_dict[key]
                )
                avg = v / float(num_iterations)
                log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = 0.0
        if global_config.precision == "fp16":
            log_string += " loss scale: {:.1f} |".format(loss_scale)
        log_string += " number of skipped iterations: {:3d} |".format(total_loss_dict[skipped_iters_key])
        log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[got_nan_key])
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory("after {} iterations".format(iteration))
            report_memory_flag = False

        timers.log(timers_to_log, normalizer=global_config.log_interval)

    return report_memory_flag


def tb_wandb_log(key, value, iteration_no, use_wandb, tensorboard_writer=None, all_ranks=False):
    # logs to both tb and wandb (if present) from the zeroth rank
    do_log = torch.distributed.get_rank() == 0 or all_ranks
    if do_log and value is not None:
        if tensorboard_writer:
            tensorboard_writer.add_scalar(key, value, iteration_no)
        if use_wandb:
            wandb.log({key: value}, step=iteration_no)


def log_norm(norm, key, iteration_no):
    if norm is not None:
        tb_wandb_log(
            key,
            norm,
            iteration_no,
            use_wandb=True,
            tensorboard_writer=None,
            all_ranks=False,
        )


COMMS_LOGGING_ENABLED = False
COMMS_LOG_RANKS = {}


def configure_deepspeed_comms_logging(
    enable=False, prof_all=False, prof_ops=None, verbose=False, debug=False, rank_logs=None
):

    global COMMS_LOGGING_ENABLED
    global COMMS_LOG_RANKS
   # comms_logger = comms.comms_logger
        
    if enable:
        comms.configure(enabled=True, verbose=verbose, debug=debug, prof_all=prof_all, prof_ops=prof_ops)
        COMMS_LOGGING_ENABLED = True
        COMMS_LOG_RANKS = set(rank_logs) if rank_logs is not None else set(range(comms.get_world_size()))
    else:
        comms.configure(enabled=False)
        COMMS_LOGGING_ENABLED = False


def enable_deepspeed_comms_logging():
    global COMMS_LOGGING_ENABLED
    comms.comms_logger.enabled = True
    COMMS_LOGGING_ENABLED = True


def disable_deepspeed_comms_logging():
    global COMMS_LOGGING_ENABLED
    comms.comms_logger.enabled = False
    COMMS_LOGGING_ENABLED = False


def serialize_comms_dict(ops=None, log_prefix="COMMS"):
    """
    Serialize collective op timings to stdout.

    This utilizes DeepSpeeds CommsLogger to record collective op latencies.

    These logs are recorded for a user-specified set of collective ops, broken down by the size of the message and a list of recorded
    values (counts, latencies, algbw, busbw).

    See https://github.com/microsoft/DeepSpeed/blob/7a5bc4fdf90d3a1cd711973ed9d0113b582f143e/deepspeed/utils/comms_logging.py#L126-L151
    """
    global COMMS_LOG_RANKS
    
    comms_logger = comms.comms_logger
    rank = comms.get_rank()
    
    should_log = rank in COMMS_LOG_RANKS
    d = comms_logger.comms_dict
    for k in d.keys():
        if ops is None or k in ops:
            for size, vals in d[k].items():
                latencies = vals[1]
                serialized_dict = "{prefix}:{op},{rank},{size},{latencies}".format(
                    prefix=log_prefix, op=k, rank=rank, size=size, latencies=",".join(str(v) for v in latencies)
                )
                if should_log:
                    print(serialized_dict)
