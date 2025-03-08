import contextlib
import itertools
import os
import pickle
import time

import torch
import torch.distributed as dist

"""
Convenience functions for logging torch CUDA Caching Allocator memory stats to wandb

 For allocations:
   memory/allocated_bytes/current
   memory/allocated_bytes/peak
   ...
   memory/reserved_bytes/current
   ...
   
For counts:
    memory/num_alloc_retries
    memory/num_sync_all_streams
    ...
"""

_MEMORY_PARENT_KEY = "memory"
_MEM_STAT_ALLOC_KEYS = [
    "allocated_bytes",  # amount of allocated memory by the torch CUDA Caching Allocator
    "reserved_bytes",  # amount of reserved memory by the torch CUDA Caching Allocator
]
_MEM_STAT_ALLOC_SUBKEYS = [
    "current",  # amount at the time of the call
    "peak",  # peak allocated (since last reset)
    "freed",  # cumulative freed
    "allocated",  # cumulative allocated
]
_MEM_STAT_COUNTER_KEYS = [
    "num_alloc_retries",  # number of failed cudaMalloc calls that result in a cache flush and retry
    "num_sync_all_streams",  # number of synchronize_and_free_events calls.
    "num_device_alloc",  # number of CUDA allocation calls. This includes both cuMemMap and cudaMalloc.
    "num_device_free",  # number of CUDA free calls. This includes both cuMemUnmap and cudaFree
]
_SUMMARY_STAT_KEYS = ["min", "median", "max", "mean", "stddev"]

MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


# These functions are mainly for debugging
def _make_alloc_keys():
    return [
        "/".join(p)
        for p in itertools.product([_MEMORY_PARENT_KEY], _MEM_STAT_ALLOC_KEYS, _MEM_STAT_ALLOC_SUBKEYS)
    ]


def _make_count_keys():
    return ["/".join(p) for p in itertools.product([_MEMORY_PARENT_KEY], _MEM_STAT_COUNTER_KEYS)]


def _flatten_dict(d, parent_key="", sep="/"):
    """
    Flattens a nested dictionary.

    Parameters:
    - d (dict): The nested dictionary to flatten.
    - parent_key (str): The base key to prepend (used in recursion).
    - sep (str): The separator between levels of the hierarchy in the flattened keys.

    Returns:
    - dict: The flattened dictionary.
    """
    flat_dict = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(_flatten_dict(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v

    return flat_dict


def format_mem_allocs(mem_allocs):
    flattened = {}
    for k in mem_allocs.keys():
        flattened.update(_flatten_dict(mem_allocs[k], parent_key=k))
    return flattened


def get_alloc_stats(stats=None, stat_keys=_MEM_STAT_ALLOC_KEYS, formatted=True):
    stats = stats or torch.cuda.memory_stats_as_nested_dict()
    # v is a dict with keys ['all', 'large', 'small'] where 'large' and 'small' are
    # different memory segments, interested in 'all'
    selected_stats = {k: v["all"] for k, v in stats.items() if k in stat_keys}
    if formatted:
        return format_mem_allocs(selected_stats)
    return selected_stats


def get_count_stats(stats=None, count_keys=_MEM_STAT_COUNTER_KEYS):
    stats = stats or torch.cuda.memory_stats_as_nested_dict()
    return {k: v for k, v in stats.items() if k in count_keys}


def get_memory_stats(include_counts=True):
    """
    Get memory stats from torch CUDA Caching Allocator

    Parameters:
    - include_counts (bool): Whether to include count stats

    Returns:
    - dict: The memory stats containing allocated bytes and reserved bytes across all memory
    segments.
    - dict: The count stats.  Cumulative counts of calls to cuda memory APIs.  See `MEM_STAT_COUNTER_KEYS`
    for queries fields.

    See https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html for details.

    """
    stats = torch.cuda.memory_stats_as_nested_dict()
    allocs = get_alloc_stats(stats)
    if include_counts:
        counts = get_count_stats(stats)
    else:
        counts = {}
    # Create flattened dict with 'memory' as the parent key
    mem_stats = _flatten_dict({**allocs, **counts}, parent_key=_MEMORY_PARENT_KEY)
    return mem_stats


def print_mem_alloc_stats(iteration=0):

    headers = ["iteration", "rank"]
    vals = [iteration, torch.distributed.get_rank()]

    for k, v in torch.cuda.memory_stats_as_nested_dict().items():
        if k in _MEM_STAT_COUNTER_KEYS:
            headers.append(k)
            vals.append(v)
    # Add prefix for easier parsing
    serialized_counts = "ALLOCATION_STATS:" + ",".join(f"{k}={v}" for k, v in zip(headers, vals))
    print(serialized_counts)
    return serialized_counts


def get_stats(keys=_MEM_STAT_COUNTER_KEYS) -> torch.Tensor:
    device = f"""cuda:{int(os.environ.get("LOCAL_RANK", "0"))}"""
    stats = torch.cuda.memory_stats()
    return torch.tensor([stats[k] for k in keys], dtype=torch.int32, device=device)


def unpack_stats(stats: torch.Tensor, keys=_MEM_STAT_COUNTER_KEYS, prefix="memory/alloc", label=""):
    return {f"{prefix}/{label}/{k}": v for k, v in zip(keys, stats.cpu().tolist())}


def calculate_summary_stats(stats_tensor: torch.Tensor):
    world_size = dist.get_world_size()
    mins = stats_tensor.min(dim=0).values
    medians = stats_tensor.median(dim=0).values
    maxs = stats_tensor.max(dim=0).values
    means = stats_tensor.sum(dim=0) / world_size
    stddev = (stats_tensor - means).pow(2).sum(dim=0).sqrt()
    return mins, medians, maxs, means, stddev


def gather_mem_alloc_stats():
    """
    Gather memory allocation counters from torch.cuda.memory_stats() across all ranks
    and return summary stats dict in wandb logging format:
    ```python
    {
        "memory/alloc/min/{key}": mins,
        "memory/alloc/median/{key}": medians,
        "memory/alloc/max/{key}": maxs,
        "memory/alloc/mean/{key}": means,
        "memory/alloc/stddev/{key}": stddev
    }
    ```
    where {key} is one of the keys in _MEM_STAT_COUNTER_KEYS
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    stats_tensor = get_stats()
    # print(f"Rank {rank} stats tensor: {stats_tensor}")
    if rank == 0:
        gathered_tensors = [torch.zeros_like(stats_tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None
    dist.gather(stats_tensor, gather_list=gathered_tensors, dst=0)
    dist.barrier()

    if rank == 0:
        gathered_stat_tensor = torch.stack(gathered_tensors)
        summary_stats = calculate_summary_stats(gathered_stat_tensor)
        stat_dict = {}
        for k, t in zip(_SUMMARY_STAT_KEYS, summary_stats):
            unpacked = unpack_stats(t, label=k)
            stat_dict.update(unpacked)
        return stat_dict
    return None


# From https://github.com/pytorch/torchtitan/blob/ba2469780da5a689e856e21ab9664ab1bed4fdd5/torchtitan/profiling.py#L75-L125
@contextlib.contextmanager
def maybe_enable_memory_snapshot(global_config, global_step: int = 0):
    enable_snapshot = global_config.enable_memory_snapshot
    rank0_only = global_config.memory_snapshot_rank0_only
    max_entries = global_config.memory_snapshot_max_entries
    freq = global_config.memory_snapshot_freq
    snapshot_dir = global_config.memory_snapshot_folder
    num_iters = global_config.memory_snapshot_num_iters
    rank = torch.distributed.get_rank()
    should_print_exit_msg = True
    
    if enable_snapshot:
        if rank0_only and rank != 0:
            return

        snapshot_dir = os.path.join(snapshot_dir, f"rank{rank}")
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int = 3):
                torch.cuda.memory._record_memory_history(max_entries=max_entries)
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1

                if not exit_ctx and (self.step_num % self.freq != 0 or self.step_num > num_iters):
                    if self.step_num > num_iters and should_print_exit_msg:
                        print(f"DEBUG::MEMORYPROFILER::rank{rank}: Reached max iterations {num_iters}. Exiting memory profiler", flush=True)
                        should_print_exit_msg = False
                    return
                
                if not exit_ctx:
                    # Subtract 1 since we step the profiler at the start of the train loop to capture memory before first iteration
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)

                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)

                msg = f"DEBUG::MEMORYPROFILER::rank{rank}: Dumping memory snapshot at step {curr_step}"
                if exit_ctx:
                    msg += f", OOM at step {self.step_num}"
                print(msg, flush=True)

                begin = time.monotonic()
                with open(f"{curr_snapshot_dir}/memory_snapshot.pickle", "wb") as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)
                msg = f"DEBUG::MEMORYPROFILER::rank{rank}: Finished dumping memory snapshot at step {self.step_num} in {time.monotonic() - begin:.2f} seconds"
                
                if exit_ctx:
                    msg += f", OOM at step {self.step_num}"
                print(msg, flush=True)

        print(f"DEBUG::MEMORYPROFILER::rank{rank}: Memory profiler active. Snapshot will be saved at {snapshot_dir}", flush=True)
        profiler = MemoryProfiler(global_step, freq)
        
        try:
            yield profiler
        except Exception as e:
            profiler.step(exit_ctx=True)
            raise e
    else:
        yield None
