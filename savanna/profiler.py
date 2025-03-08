
import os
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple
import shutil
import torch
import torch.distributed

from omegaconf import DictConfig
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import tensorboard_trace_handler
from .logging import get_world_size_and_rank, logger


def _warn(msg: str):
    _, rank = get_world_size_and_rank()
    if rank == 0:
        logger.warning(msg)


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str,
    metric: str = "self_cuda_time_total",
    row_limit: int = 100,
):
    """
    Handles export of artifacts from `torch.profiler.profile`.

    The following artifacts are exported:
    - chrome / tensorboard trace - viewable through tensorboard or perfetto.dev / chrome::/tracing
    - trace event table
    - memory timeline if `profile_memory`
    - stacks if `with_stack` (note that `profile_memory` requires `with_stack` to be `True`),
    viewable as a flamegraph see (https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

    Notes:
    - Each profiling cycle is exported as a sub-directory in output_dir
        - E.g., profiling in 5-step cycle (wait=2, warmup=2, active=1, repeat=0) will result in
        sub-directories iteration_5, iteration_10, etc.
    - If profiling in a distributed setting, each artifact will be prefixed with rank.
    - Memory timeline is only exported for rank 0 (error if exporting from multiple ranks on single node)

    See profiler documentation (https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for more details

    Args:
        prof (torch.profiler.profile): instance of torch profiler to use
        output_dir (str):  directory to store artifacts
        metric (str): metric to order trace event table by, see `torch.profiler.profile.key_averages().table` for
        row_limit (int): number of rows to display in trace event table

    """
    world_size, rank = get_world_size_and_rank()
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    if rank == 0:
        logger.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces with prof.export_chrome_trace
    exporter = tensorboard_trace_handler(curr_trace_dir, worker_name=f"rank{rank}", use_gzip=True)
    exporter(prof)

    if rank == 0:
        logger.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if prof.profile_memory:
        if rank == 0:
            try:
                prof.export_memory_timeline(f"{curr_trace_dir}/rank{rank}_memory-timeline.html")
            except Exception as e:
                logger.warn(f" Failed to export memory timeline: {e}")

    # Dump stack traces
    if prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    key_avgs = prof.key_averages(group_by_input_shape=prof.record_shapes, group_by_stack_n=5).table(
        sort_by=metric, row_limit=row_limit
    )
    with open(f"{curr_trace_dir}/rank{rank}_key_averages.txt", "w") as f:
        print(key_avgs, file=f)
    if rank == 0:
        logger.info(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    # if world_size > 1:
    #     torch.distributed.barrier()


class DummyProfiler:
    """
    Drop-in replacement for torch.profiler.profile that functions as a nullcontext / object
    with no-op methods for `start`, `stop`, and `step`.

    This is helpful for instrumenting profiling in a recipe without requiring changes to the
    code independent of whether profiling is on / off.

    E.g.,
    ```
        profiler = DummyProfiler()
        #profiler = torch.profiler.profile()

        # Below is same regardless of profiler object type
        with profiler as prof:
            for epoch in epochs:
                for batch in batches:
                    train.step()
                    prof.step()
    ```
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def step(self):
        pass


def setup_torch_profiler(
    enable: bool = False,
    cpu: bool = True,
    cuda: bool = True,
    profile_memory: bool = False,
    with_stack: bool = False,
    record_shapes: bool = True,
    with_flops: bool = False,
    wait: int = 5,
    warmup: int = 5,
    active: int = 2,
    repeat: int = 1,
    output_dir: str = "torchprofiler_traces",
    clean_output_dir: bool = False,
    num_rows: int = 100,
) -> Tuple[torch.profiler.profile, DictConfig]:
    """

    Configures `torch.profiler` for exporting traces, memory timeline, stack traces, etc.

    See `trace_handler` for more details on exported artifacts.

    NOTE:
        - Enabling the profiler will result in training speed reduction.
        - Setting `profile_memory: True` will generate large trace files.

    Args:
        enabled (bool): Enable pytorch profiler. Default is False.
        cpu (bool): Enable cpu profiling. Default is True.
        cuda (bool): Enable cuda profiling. Default is True.
        profile_memory (bool): Profile memory usage. Default is False.
        with_stack (bool): Profile stack. Default is False.
        record_shapes (bool): Record shapes. Default is False.
        with_flops (bool): Profile flops. Default is False.
        wait (Optional[int]): Wait time in steps. Maps to `wait` kwarg of `torch.profiler.schedule`.
        warmup (Optional[int]): Warmup time in steps. Maps to `warmup` kwarg of `torch.profiler.schedule`.
        active (Optional[int]): Active time in steps. Maps to `active` kwarg of `torch.profiler.schedule`.
        repeat (Optional[int]): Number of profiling cycles.  Maps to `repeat` kwarg of `torch.profiler.schedule`.
        output_dir (Optional[str]): Tracing file output path.
    """

    if not enable:
        _warn(" Profiling disabled.")

        return DummyProfiler(), DictConfig({"enabled": False})

    # Set up profiler activities
    activities = []
    if cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    if profile_memory:
        _warn(
            "`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True"
        )
    with_stack = with_stack or profile_memory
    record_shapes = record_shapes or profile_memory
    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    if clean_output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    # trace_handler manages the export of profiler artifacts
    # this callback will be triggered after **each** profiling cycle
    callback = partial(trace_handler, output_dir=output_dir, row_limit=num_rows)

    profiler = torch.profiler.profile(
        activities=activities,
        profile_memory=profile_memory,
        with_stack=with_stack,
        record_shapes=record_shapes,
        with_flops=with_flops,
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=callback,
    )
    rank, _ = get_world_size_and_rank()

    if rank == 0:
        logger.info("Profiler setup complete.")

    return profiler
