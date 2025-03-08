import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import tensorboard_trace_handler

from savanna.arguments.global_config import GlobalConfigProfiler

from .logging import get_world_size_and_rank, logger


def _warn(msg: str):
    _, rank = get_world_size_and_rank()
    if rank == 0:
        logger.warning(msg)


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str,
    metric: str = "self_cuda_time_total",
    row_limit: int = -1,
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
    #torch.save(prof.key_averages(), f"{curr_trace_dir}/rank{rank}_key_averages.pt")
    
    if rank == 0:
        logger.info(f"Saving profiling results to {curr_trace_dir}")

    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    if world_size > 1:
        torch.distributed.barrier()


class BaseProfiler:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def step(self):
        pass

    @contextmanager
    def mark(self, name):
        raise NotImplementedError


class EmptyProfiler(BaseProfiler):
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

    def __init__(self):
        pass

    @contextmanager
    def mark(self, name):
        yield


class TorchProfiler(BaseProfiler):
    """
    Configures `torch.profiler` for exporting traces, memory timeline, stack traces, etc.

    See `trace_handler` for more details on exported artifacts.

    NOTE:
        - Enabling the profiler will result in training speed reduction.
        - Setting `profile_memory: True` will generate large trace files.

    Args:
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

    def __init__(
        self,
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
    ):
        activities = []
        self.cpu = cpu
        self.cuda = cuda

        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.with_flops = with_flops

        self.output_dir = output_dir
        self.clean_output_dir = clean_output_dir
        self.num_rows = num_rows
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat

        if cpu:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.schedule = torch.profiler.schedule(
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
            schedule=self.schedule,
            experimental_config=experimental_config,
            on_trace_ready=callback,
        )
        rank, _ = get_world_size_and_rank()

        if rank == 0:
            logger.info("Profiler setup complete.")

        self.profiler = profiler

    def __enter__(self):
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self):
        self.profiler.step()

    @contextmanager
    def mark(self, name):
        with torch.profiler.record_function(name):
            yield

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=10):
        return self.profiler.key_averages(
            group_by_input_shape=group_by_input_shape, group_by_stack_n=group_by_stack_n
        )

    def events(self):
        return self.profiler.events()


class CudaProfiler(BaseProfiler):
    """
    Profiling context for use with `nsys`

    Example:
    - Use as context manager:
        ```
            with CudaProfilerCtx():
                model.train()
        ```
    - Use as object:
        ```
        profiler = CudaProfilerCtx()
        profiler.start()
        model.train()
        profiler.stop()
        ```
    """

    def __init__(
        self,
        warmup_steps=3,
        num_steps=2,
        emit_nvtx=True,
        disable_autograd_multithreading=True,
        stop_on_exit=True,
        debug=False,
    ):
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.total_steps = warmup_steps + num_steps
        self.step_count = 0
        self.active = False
        self.nvtx = None
        self.debug = debug

        if emit_nvtx:
            self.nvtx = torch.autograd.profiler.emit_nvtx()

        self.disable_autograd_multithreading = disable_autograd_multithreading
        multithreading_enabled = not self.disable_autograd_multithreading
        torch.autograd.set_multithreading_enabled(multithreading_enabled)

        self.stop_on_exit = stop_on_exit

    def should_profile(self, step_count):
        if self.debug:
            print(
                f"Should profile: {self.warmup_steps} <= {step_count} < {self.total_steps} = {self.warmup_steps <= step_count < self.total_steps}"
            )
        return self.warmup_steps <= step_count < self.total_steps

    def step(self):
        if self.debug:
            print("Step", self.step_count, "Active", self.active)
        self.step_count += 1

        # Only case when we should be profiling
        if self.should_profile(self.step_count):
            if not self.active:
                self.active = True
                self.__enter__()
        else:
            if self.active:
                self.active = False
                self.__exit__(None, None, None)

    def start(self):
        #
        pass
        # if not self.active:
        #     self.active = True
        #     self.__enter__()

    def stop(self):
        if self.active:
            self.__exit__(None, None, None)

    def __enter__(self):
        if self.should_profile(self.step_count):
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print("Starting cuda profiler")
            else:
                print("Starting cuda profiler")

            if self.nvtx is not None:
                self.nvtx.__enter__()

            torch.cuda.profiler.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print("Stopping cuda profiler")
        else:
            print("Stopping cuda profiler")

        if self.nvtx is not None:
            self.nvtx.__exit__(exc_type, exc_value, exc_traceback)

        # torch.autograd.set_multithreading_enabled(True)

        if self.stop_on_exit:
            torch.cuda.profiler.stop()

        if exc_type is not None:
            print(f"Exception occurred: {exc_type}, {exc_value}")

        # return True

    @contextmanager
    def mark(self, name):
        with torch.cuda.nvtx.range(name):
            yield


def setup_profiler(global_config: GlobalConfigProfiler):

    if not global_config.should_profile or global_config.profiler_type == "none":   
        return EmptyProfiler()
    
    profiler_type = global_config.profiler_type
    if profiler_type == "nsys":
        profiler = CudaProfiler(
            num_steps=global_config.nsys_num_steps,
            warmup_steps=global_config.nsys_warmup_steps,
            emit_nvtx=global_config.emit_nvtx,
            disable_autograd_multithreading=global_config.disable_autograd_multithreading,
            stop_on_exit=global_config.nsys_stop_on_exit,
        )
    elif profiler_type == "torch":
        rank = torch.distributed.get_rank()
        profile_ranks = global_config.profile_ranks
        if profile_ranks is not None:
            rank_should_profile = rank in profile_ranks
        else:
            rank_should_profile = True 
            
        if rank_should_profile:
            profiler = TorchProfiler(
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
        else:
            profiler = EmptyProfiler()
    else:
        raise NotImplementedError(
            f"Unknown profiler {profiler_type}, supported profilers are `nsys` and `torch`"
        )

    return profiler


def run_script(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Print the output from stdout in real time
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")  # end='' prevents double newlines

        # Wait for the process to finish and get the exit code
        process.stdout.close()
        process.wait()

        # Optionally, print the stderr after the process completes
        stderr_output = process.stderr.read()
        if stderr_output:
            print("Subprocess STDERR:\n", stderr_output)

        # Check the exit code
        if process.returncode != 0:
            print(f"Subprocess failed with return code {process.returncode}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        process.stdout.close()
        process.stderr.close()


class NsysProfiler:
    """
    Self-contained `nsys` runner.

    Runs `nsys` for a given script and outputs optional stats report.

    See `tests/functional/test_profiler.py` for example usage.

    """

    def __init__(
        self,
        trace_dir="traces",
        gpu_metrics_device="all",
        cuda_memory_usage=True,
        cudabacktrace=True,
        python_sampling=True,
        capture_range="cudaProfilerApi",
        stats=False,
        nic_metrics=True,
        show_output=True,
        trace="cuda,nvtx,osrt,cudnn,cublas-verbose",
        sample="process-tree",
        output_file_prefix="profile-%h-%p",  # %h = hostname, %p = pid
        force_overwrite=True,
        stop_on_exit=True,
        inherit_environment=True,
        append_timestamp=False,
        clean_output_dir=False,
    ):
        self.trace_dir = trace_dir
        self.gpu_metrics_device = gpu_metrics_device
        self.cuda_memory_usage = cuda_memory_usage
        self.cudabacktrace = cudabacktrace
        self.python_sampling = python_sampling
        self.capture_range = capture_range
        self.stats = stats
        self.nic_metrics = nic_metrics
        self.show_output = show_output
        self.trace = trace
        self.sample = sample
        self.output_file_prefix = output_file_prefix
        self.force_overwrite = force_overwrite
        self.stop_on_exit = stop_on_exit
        self.inherit_environment = inherit_environment
        # Create timestamp for unique trace file output
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure the trace directory exists
        self.output_path = (
            os.path.join(self.trace_dir, self.timestamp, self.output_file_prefix)
            if append_timestamp
            else os.path.join(self.trace_dir, self.output_file_prefix)
        )

        if clean_output_dir:
            shutil.rmtree(self.trace_dir, ignore_errors=True)

        os.makedirs(self.trace_dir, exist_ok=True)

    def _bool_to_str(self, bool_val):
        return "true" if bool_val else "false"

    def export_report(
        self, report_path, stats_report="nvtx_pushpop_trace", report_prefix="stats", report_format="csv"
    ):
        print(f"Report path: {report_path}")
        stats_output_prefix = os.path.join(self.trace_dir, report_prefix)
        stats_cmd = [
            "nsys",
            "stats",
            f"--format={report_format}",
            f"--output={stats_output_prefix}",
            f"--report={stats_report}",
            report_path.as_posix(),
        ]
        stats_script = os.path.join(self.trace_dir, "nsys_stats.sh")

        self.write_script(stats_cmd, stats_script)
        print("Running command:", " ".join(stats_cmd))
        run_script(stats_script)

        stats_path = [p for p in Path(self.trace_dir).rglob(f"{report_prefix}*.{report_format}")][0]
        print(f"Stats path: {stats_path}")

        return stats_path.absolute().as_posix()

    def write_script(self, cmd, script_path):
        print(f"Script path: {script_path}")
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)
        os.chmod(script_path, 0o777)

    def run(self, script: list[str], script_args: list[str] = None, stats_report="nvtx_pushpop_trace"):
        if script_args is None:
            script_args = []

        # Build the nsys command
        nsys_cmd = [
            "nsys",
            "profile",
            f"--gpu-metrics-device={self.gpu_metrics_device}",
            f"--cuda-memory-usage={self._bool_to_str(self.cuda_memory_usage)}",
            f"--cudabacktrace={self._bool_to_str(self.cudabacktrace)}",
            f"--python-sampling={self._bool_to_str(self.python_sampling)}",
            f"--capture-range={self.capture_range}",
            f"--stats={self._bool_to_str(self.stats)}",
            f"--nic-metrics={self._bool_to_str(self.nic_metrics)}",
            f"--show-output={self._bool_to_str(self.show_output)}",
            f"--trace={self.trace}",
            f"--sample={self.sample}",
            f"--output={self.output_path}",
            f"--force-overwrite={str(self.force_overwrite).lower()}",
            f"--stop-on-exit={str(self.stop_on_exit).lower()}",
            f"--inherit-environment={str(self.inherit_environment).lower()}",
        ]

        parts = [" ".join(l) for l in [nsys_cmd, script, script_args]]
        cmd = " ".join(parts)
        print("Running command:", cmd)

        script_path = os.path.join(self.trace_dir, "nsys_script.sh")
        self.write_script(cmd, script_path)
        run_script(script_path)

        if stats_report is not None:
            report_path = next((Path(self.trace_dir).rglob("*.nsys-rep")))
            stats_path = self.export_report(report_path, stats_report=stats_report)
        else:
            return report_path
        return report_path, stats_path
