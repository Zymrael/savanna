"""Pretrain utilities."""

import gc
import logging
import math
import os
import sys
import time
from concurrent.futures import wait
from functools import partial
from pathlib import Path

import deepspeed
import torch
from deepspeed.utils import (
    set_z3_leaf_modules,
)

from savanna import mpu, print_datetime, print_rank_0
from savanna.checkpointing import (
    delete_queue,
    finalize_async_save,
    load_checkpoint,
    read_global_step,
    save_checkpoint,
)
from savanna.data.data_utils import build_train_valid_test_data_iterators
from savanna.debug import save_tensor_hook
from savanna.initialize import initialize_megatron
from savanna.learning_rates import AnnealingLR
from savanna.logging import comms as ds_comms
from savanna.logging import (
    configure_deepspeed_comms_logging,
    disable_deepspeed_comms_logging,
    enable_deepspeed_comms_logging,
    get_total_flops,
    serialize_comms_dict,
    tb_wandb_log,
    training_log,
)
from savanna.memory_stats import print_mem_alloc_stats
from savanna.mfu import add_model_flop_utilization_inputs
from savanna.model import (
    BackbonePipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
    mark_norms_for_sequence_parallel_grad_sync,
)
from savanna.model.backbone import (
    cross_entropy,
    dpo_loss,
    oadm_loss,
    reweighted_cross_entropy,
)
from savanna.optimizers.sophia import SophiaG
from savanna.profiler import BaseProfiler, setup_profiler
from savanna.tokenizer import CharLevelTokenizer
from savanna.utils import (
    CharCounter,
    NullDetector,
    OverflowMonitor,
    Timers,
    get_diffusion_mask,
    get_ltor_masks_and_position_ids,
    get_mlm_masks,
    get_noise_scale_logger,
    get_span_masks,
    get_streams,
    get_total_params,
    init_wandb,
    make_upper_case,
    mask_control_tags,
    print_straggler_report,
    reduce_losses,
)

logger = logging.getLogger(__name__)

hsd_timer = NullDetector()
straggler = None

real_empty_cache = torch.cuda.empty_cache
torch.cuda.empty_cache = lambda: None
RealEvent = torch.cuda.Event


class RecycleableEvent:
    events = set()

    """Autorecycles on __del__"""

    def __init__(self, *args, **kwargs):
        try:
            event = self.events.pop()
        except KeyError:
            event = RealEvent(*args, **kwargs)
        self.event = event

    def record(self, *args, **kwargs):
        return self.event.record(*args, **kwargs)

    def synchronize(self):
        return self.event.synchronize()

    def query(self, *args, **kwargs):
        return self.event.query(*args, **kwargs)

    def __del__(self):
        self.events.add(self.event)

    def wait(self, *args, **kwargs):
        return self.event.wait(*args, **kwargs)


def monkey_patch_event():
    torch.cuda.Event = torch.cuda.streams.Event = RecycleableEvent
    from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator

    CUDA_Accelerator.Event = RecycleableEvent

    assert os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS") is None
    assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") is None


class AllocStats:
    """Compares allocation stats between steps, and Prints on rank 0 when allocations have happened."""

    def __init__(self, rank, global_config):
        self.rank = rank
        self.last_stats = None
        self.global_config = global_config
        
    def step(self, iteration):
        s = torch.cuda.memory_stats()
        stats = {
            "num_alloc_retries": s["num_alloc_retries"],
            "num_device_alloc": s["num_device_alloc"],
            "num_device_free": s["num_device_free"],
        }
        if iteration == 6:
            torch.cuda.empty_cache()
        if iteration > 7 and stats != self.last_stats and self.global_config.print_mem_alloc_stats:
            logger.warning(
                f"[rank={self.rank}] [gdb] Allocation stats changed between iteration {iteration-1} and {iteration}. This isn't necessarily a problem, but generally indicates high memory pressure and can result in the job running slower: {self.last_stats} -> {stats}. If this warning becomes annoying, feel free to just disable."
            )

        # Should be false
        if self.global_config.prealloc_mem:
            if iteration == 10:
                # Compute amount of free memory for each stream, using torch.cuda.memory_snapshot()
                alloc = {}
                for segment in torch.cuda.memory_snapshot():
                    assert segment['is_expandable'], 'Looks like expandable_segments is not enabled. Please re-launch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set'
                    stream = segment['stream']
                    alloc[stream] = alloc.get(stream, 0) + segment['total_size']

                real_empty_cache()

                free = {}
                for segment in torch.cuda.memory_snapshot():
                    stream = segment['stream']
                    spare = segment['total_size'] - segment['allocated_size']
                    free[stream] = free.get(stream, 0) + spare

                free_memory = torch.cuda.mem_get_info()[0]
                targets = {}
                for stream, allocated in alloc.items():
                    if stream == 0:
                        continue
                    targets[stream] = min(2 * allocated, allocated + int(3e9))
                buffer_size = 500 * 1024 * 1024 + 200 * 2**20 * (len(targets) + 1)
                targets[0] = free_memory - sum(targets.values()) - buffer_size

                streams = get_streams()
                print(f'{free.keys()=} {[s.cuda_stream for s in streams]}')
                # Allocate free ram equally across streams
                alloc_size = free_memory - buffer_size
                print(f'Allocating buffer: {alloc_size=} across {len(streams)} streams ({targets})')
                allocations = []
                for stream in streams:
                    if stream.cuda_stream not in targets:
                        continue
                    with torch.cuda.stream(stream):
                        # Make lots of small segments, they are cheap but we can need a lot of them
                        for i in range(2*100):
                            allocations.append(torch.empty(2**20, device='cuda', dtype=torch.uint8))
                        if alloc_size > 0:
                            target = targets.get(stream.cuda_stream, 0)
                            allocations.append(torch.empty(target + free.get(stream.cuda_stream, 0), device='cuda', dtype=torch.uint8))
                allocations.clear()
                if self.rank == 0:
                    # bp()
                    pass
                s = torch.cuda.memory_stats()
                stats = {
                    'num_alloc_retries': s['num_alloc_retries'],
                    'num_device_alloc': s['num_device_alloc'],
                    'num_device_free': s['num_device_free'],
                }
                os.environ['DISABLE_MALLOC'] = '1'
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
    print("base shapes saved...exiting")
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


def update_data_loading_config(global_config):
    """Load checkpointed state related to data loading."""
    assert global_config.load is not None
    assert global_config.iteration is not None

    iteration = global_config.iteration

    if global_config.checkpoint is not None and \
       global_config.checkpoint.get("load_universal", False):
        tag = f"global_step{iteration}_universal"
    else:
        tag = f"global_step{iteration}"

    ckpt_dir = f"{global_config.load}/{tag}"
    # Rank 0 model states should always be saved.
    if global_config.zero_optimization["stage"] > 1:
        state_dict = torch.load(f"{ckpt_dir}/zero_pp_rank_0_mp_rank_00_model_states.pt", map_location=torch.device("cpu"), weights_only=False)
    else:
        state_dict = torch.load(f"{ckpt_dir}/mp_rank_00_model_states.pt", map_location=torch.device("cpu"), weights_only=False)

    # Set previous data loading values.
    if "data_loading" in state_dict:
        checkpoint_data_loading = state_dict["data_loading"]
        if global_config.train_data_token_index is None:
            global_config.train_data_token_index = checkpoint_data_loading.get(
                "train_data_token_index"
            )
            print(f"LOAD_TRAIN_DATA_TOKEN_INDEX: Setting train_data_token_index to {global_config.train_data_token_index}")
        if global_config.use_checkpoint_num_samples:
            global_config.train_val_test_num_samples = checkpoint_data_loading.get(
                "train_val_test_num_samples"
            )
            print(f"LOAD_TRAIN_VAL_TEST_NUM_SAMPLES: Setting train_val_test_num_samples to {global_config.train_val_test_num_samples}")


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
    print_datetime("Starting setup", rank_0=False)
    
    # setup logging and timers
    init_wandb(global_config=global_config)
    timers = Timers(
        use_wandb=global_config.use_wandb,
        tensorboard_writer=global_config.tensorboard_writer,
    )

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(global_config=global_config)

    rank = torch.distributed.get_rank()
    
    if rank == 0:
        NCCL_DEBUG = os.environ.get("NCCL_DEBUG", None)
        print(f"NCCL_DEBUG {NCCL_DEBUG}", flush=True)
        dp_group = mpu.get_data_parallel_group()
        print(f"DP Group {torch.distributed.get_process_group_ranks(dp_group)}")
        mp_group = mpu.get_model_parallel_group()
        print(f"MP Group {torch.distributed.get_process_group_ranks(mp_group)}")
        
        AVOID_RECORD_STREAMS = os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS", None)
        print(f"AVOID_RECORD_STREAMS {AVOID_RECORD_STREAMS}")
        cca_cfg = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)
        print(f"PYTORCH_CUDA_ALLOC_CONFIG {cca_cfg}")
        
        if global_config.expandable_segments:
            assert cca_cfg is not None
            assert "expandable_segments:True" in cca_cfg

    # TransformerEngine Env Attn Vars
    if global_config.recycle_events and global_config.use_cp_flash_te:
        assert os.environ.get("NVTE_TORCH_COMPILE", "1") == "0", "NVTE_TORCH_COMPILE must be disabled when using flash_te and recycling events"
        assert os.environ.get("PYTORCH_JIT", "1") == "0", "PYTORCH_JIT must be disabled when using flash_te and recycling events"
    
    # These should be exported by the launcher script
    if global_config.use_cp_flash_te:
        assert os.environ["NVTE_FLASH_ATTN"] == "1", "NVTE_FLASH_ATTN must be enabled when use_cp_flash_te = True"
        assert os.environ["NVTE_FUSED_ATTN"] == "0", "NVTE_FUSED_ATTN must be disabled when use_cp_flash_te = True"
        assert os.environ["NVTE_UNFUSED_ATTN"] == "0", "NVTE_UNFUSED_ATTN must be disabled when use_cp_flash_te = True"

    # if global_config.te_attn_backend is not None:
    #     if global_config.te_attn_backend == "FLASH":
    #         os.environ["NVTE_FLASH_ATTN"] = "1"
    #         os.environ["NVTE_FUSED_ATTN"] = "0"
    #         os.environ["NVTE_UNFUSED_ATTN"] = "0"
    #     elif global_config.te_attn_backend == "FUSED":
    #         os.environ["NVTE_FLASH_ATTN"] = "0"
    #         os.environ["NVTE_FUSED_ATTN"] = "1"
    #         os.environ["NVTE_UNFUSED_ATTN"] = "0"
    #     elif global_config.te_attn_backend == "UNFUSED":
    #         os.environ["NVTE_FLASH_ATTN"] = "0"
    #         os.environ["NVTE_FUSED_ATTN"] = "0"
    #         os.environ["NVTE_UNFUSED_ATTN"] = "1"
    #     else:
    #         raise ValueError(f"Invalid te_attn_backend: {global_config.te_attn_backend}")
    
    if global_config.nvte_debug:
        os.environ["NVTE_DEBUG"] = "1"
        os.environ["NVTE_DEBUG_LEVEL"] = str(global_config.nvte_debug_level)
        
    # TransformerEngine Attention Env Vars
    if rank == 0:
        deterministic = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
        nvte_flash = os.environ.get("NVTE_FLASH_ATTN", None)
        nvte_fused = os.environ.get("NVTE_FUSED_ATTN", None)
        nvte_flash_bwd = os.environ.get("NVTE_FUSED_ATTN_USE_FAv2_BWD", None)
        nvte_debug = os.environ.get("NVTE_DEBUG", None)
        nvte_debug_level = os.environ.get("NVTE_DEBUG_LEVEL", None)
        print(f"DEBUG::TRANFORMERENGINE::NVTE_ALLOW_NONDETERMINISTIC_ALGO: {deterministic}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_FLASH_ATTN: {nvte_flash}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_FUSED_ATTN: {nvte_fused}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_FUSED_ATTN_USE_FAv2_BWD: {nvte_flash_bwd}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_DEBUG: {nvte_debug}")
        print(f"DEBUG::TRANFORMERENGINE::NVTE_DEBUG_LEVEL: {nvte_debug_level}")

    # Ensure triton cache is per process 
    # triton_cache = os.environ.get("TRITON_CACHE_DIR", None)
    # from triton.runtime.cache import FileCacheManager
    # fm = FileCacheManager(key="savanna")
    # print(f"TRITON_CACHE_CHECK: rank{rank}: {triton_cache=} {fm.cache_dir=}")

    # if global_config.load is not None, there are 2 cases: 
    # 1) iteration should be already be set by user-provided model_config
    # 2) a load path is provided by user
        # load_path contains a "latest" file with the iteration number, which is the format when saving models with Deepspeed
        # load_path does NOT contain a "latest" file, in which case we default to iteration 0.  This will be the case when
        # the user wishes to resubmit the same model_config to SLURM with the iteration being set dynamically during runtime
        # based on the checkpoint
        # When first starting training, the checkpoint path will be empty
        # Upon restarts (assuming a checkpoint has already been saved), the iteration will be read from "latest"

    if global_config.load is None or global_config.finetune:
        global_config.iteration = 0
        if global_config.load is not None and global_config.finetune:
            load_iteration = read_global_step(global_config.load)
            print_rank_0(f"LOAD_CHECKPOINT: Finetuning from iteration {global_config.iteration} from checkpoint iteration {load_iteration}")
        else:
            load_iteration = 0
    else:    
        # Handle the case when load is provided but iteration is not
        # In this case, the load directory should be in the expected DS format, containing a "latest" file
        # with the iteration number.
        if global_config.iteration is None:
            load_iteration = global_config.iteration = read_global_step(global_config.load)
        else:
            load_iteration = global_config.iteration
        
        # These should be set to True if we are resuming from a checkpoint (resuming training) but not when finetuning
        if global_config.iteration > 0 and not global_config.finetune:
            checkpoint_lr_scheduler = global_config.use_checkpoint_lr_scheduler
            if not checkpoint_lr_scheduler:
                print_rank_0("LOAD_CHECKPOINT `checkpoint_lr_scheduler` is False even though resuming from checkpoint, setting to True")
                global_config.use_checkpoint_lr_scheduler = True
            
            checkpoint_num_samples = global_config.use_checkpoint_num_samples
            if not checkpoint_num_samples:
                print_rank_0("LOAD_CHECKPOINT `checkpoint_num_samples` is False even though resuming from checkpoint, setting to True")
                global_config.use_checkpoint_num_samples = True

            # Update data load index, only loading from checkpoint
            update_data_loading_config(global_config)

    # Additional check to ensure iteration is set correctly
    # We need to distinguish between global_config.iteration and load_interaction
    # The first is needed for correctly setting data loading indices
    # The latter is needed for loading from the correct checkpoint
    # When finetuning, the global_config.iteration = 0 and load_iteration is set to the checkpoint iteration
    # When resuming training (i.e., after restart), the two are equal, the latest checkpointed iteration.

    print_rank_0(f"LOAD_CHECKPOINT: Initialized global config iteration to {global_config.iteration} and load iteration to {load_iteration}")
    
    print_datetime("Building data iterators", rank_0=True)
    # Data stuff.
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
        valid_data_iterator_list,
    ) = build_train_valid_test_data_iterators(global_config=global_config)
    timers("train/valid/test data iterators").stop()

    print_datetime("Setting up model and optimizer", rank_0=True)
    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(global_config=global_config, use_cache=False, iteration=load_iteration)
    timers("model and optimizer").stop()
    
    
    if global_config.debug_dir is not None:
        print_rank_0("Registering debugging tensor hooks")
        for name, module in model.sequential.named_modules():
            save_tensor_hook(name, module, global_config) 
    
    
    print(f"GLOBAL ITERATION AFTER MODEL SETUP: {global_config.iteration}")
    if global_config.use_mup and global_config.coord_check:
        mup_coord_check(global_config, timers, lr_scheduler, train_data_iterator)

    # Calculate model flops for MFU logging
    add_model_flop_utilization_inputs(global_config=global_config)

    global hsd_timer
    if global_config.heimdall_log_straggler:
        try:
            from heimdall.straggler.detector import StragglerDetector
            hsd_timer = StragglerDetector()
            print("heimdall.straggler.detector imported")
        except ImportError:
            print("Unable to import heimdall.straggler.detector")

    if global_config.heimdall_log_straggler and not isinstance(hsd_timer, NullDetector):
        assert global_config.heimdall_log_interval % global_config.log_interval == 0, "`hsd` log interval must be multiple of `log_interval`"
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = global_config.heimdall_straggler_minmax_count
        should_enable = not global_config.heimdall_disable_straggler_on_startup
        port = global_config.heimdall_straggler_port
        print(f"Configuring heimdall.straggler.detector with {mmcnt=} {port=}")
        hsd_timer.configure(world, rank, mmcnt=mmcnt, enabled=should_enable, port=port)
    
    global straggler
    if global_config.use_next:
        try:
            import nvidia_resiliency_ext.straggler as straggler
        except ImportError as e:
            print("Unable to import nvidia_resiliency_ext.straggler", e)
        else:
            print("Initializing straggler.Detector")
            scores = ["individual_perf_scores"]
            if global_config.next_gather_on_rank0:
                scores += ["relative_perf_scores"] 
            straggler.Detector.initialize(
            scores_to_compute=scores,
            gather_on_rank0=global_config.next_gather_on_rank0 # all ranks results will be available on rank 0
        )
        torch.distributed.barrier()    
    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(["model and optimizer", "train/valid/test data iterators"])
    print_rank_0("training ...")
    print_datetime("Starting training", rank_0=True)

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
            valid_data_iterator_list=valid_data_iterator_list,
        )

    if straggler is not None:
        straggler.Detector.shutdown()

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
    
    if global_config.async_save:
        finalize_async_save(global_config, blocking=True)
        
    # To ensure checkpoint files are uploaded to s3.
    # rilango - fix crashes when save is not set

    if global_config.checkpoint_stores is not None:
        upload_indicator = Path(os.path.join(global_config.save, '.uploading'))
        while os.path.exists(upload_indicator) and len(os.listdir(upload_indicator)) > 0:
            print_rank_0("Checkpoint is still getting uploaded. Please wait...")
            time.sleep(10)

    global delete_queue
    wait(delete_queue)


def _get_batch_dpo(global_config, tokenizer, keys, data, datatype):
    """
    A data batch for DPO is formatted in a particular way, dictated by
    `tools/prepropress_data_dpo.py`.

    This function splits the DPO data by pad tokens and prepares it for `dpo_loss`.
    Of note, the returned `labels` contain both the token ids and, in the last
    column, the reference loprobs.
    """
    if mpu.get_sequence_parallel_world_size() > 1:
        raise ValueError("Context parallelism not supported")
    
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
    if mpu.get_sequence_parallel_world_size() > 1:
        raise ValueError("Context parallelism not supported")

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
    if mpu.get_sequence_parallel_world_size() > 1:
        raise ValueError("Context parallelism not supported")

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
    if mpu.get_sequence_parallel_world_size() > 1:
        raise ValueError("Context parallelism not supported")

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
    return (
        mpu.zigzag_split_across_cp_ranks(tokens),
        mpu.zigzag_split_across_cp_ranks(labels),
        mpu.zigzag_split_across_cp_ranks(loss_mask),
        mpu.zigzag_split_across_cp_ranks(attention_mask, -2)
        if global_config.materialize_attn_mask
        else None,
        mpu.zigzag_split_across_cp_ranks(position_ids),
    )


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
    
    with hsd_timer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            global_config=global_config, data_iterator=data_iterator
        )

    if global_config.to_upper in ("upper", "weighted", "normalized_weighted"):
        tokens, _ = make_upper_case(tokens) # Make all tokens uppercase before passing to the model. Labels are handled by loss
        if global_config.to_upper == "upper":
            labels = make_upper_case(labels)


    if timers is not None:
        timers("batch generator").stop()

    with hsd_timer:
        outputs = model((tokens, position_ids, attention_mask), global_config=global_config)
   
    rank = torch.distributed.get_rank()
    if (
        is_train
        and global_config.curriculum_learning
        and global_config.curriculum_seqlen < global_config.seq_length
    ):
        loss_mask = loss_mask[:, : global_config.curriculum_seqlen].contiguous()
        labels = labels[:, : global_config.curriculum_seqlen].contiguous()
    
    
    if global_config.mask_loss_control_tags:
        loss_mask = mask_control_tags((labels, loss_mask), global_config.tokenizer.eod)
    
    if global_config.alignment_method is None:
        if global_config.pretraining_strategy == "OADM":
            loss = oadm_loss(outputs, (labels, loss_mask), _fp16=global_config.fp16_lm_cross_entropy)
        else:
            if global_config.to_upper == "weighted":
                loss = reweighted_cross_entropy(outputs, (labels, loss_mask), _fp16=global_config.fp16_lm_cross_entropy, lowercase_weight=global_config.lowercase_loss_reweighting, normalize_per_batch=False)
                return loss

            elif global_config.to_upper == "normalized_weighted":
                loss = reweighted_cross_entropy(outputs, (labels, loss_mask), _fp16=global_config.fp16_lm_cross_entropy, lowercase_weight=global_config.lowercase_loss_reweighting, normalize_per_batch=True)
                return loss

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
            if "soft_embedding" not in name:
                param.requires_grad = False

    ### LoRA stuff ###
    if global_config.lora is not None and global_config.lora.get("enabled", False):
        from savanna.lora import apply_lora_to_model

        apply_lora_to_model(model, global_config)
        if global_config.lora.get("freeze_base_model", True):
            modules_to_save = global_config.lora.get("modules_to_save") or []
            for name, param in model.named_parameters():
                # Use leaf-name matching (consistent with apply_lora_to_model) to
                # avoid substring false positives (e.g. "norm" matching "layernorm").
                leaf = name.rsplit(".", 1)[-1]
                if "lora_" not in name and not any(m == leaf for m in modules_to_save):
                    param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print_rank_0(f"LoRA: {trainable:,} trainable params / {total:,} total ({100 * trainable / total:.2f}%)")

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

    deepspeed_config = global_config.deepspeed_config
    use_zero_3 = global_config.zero_optimization["stage"] == 3
    hpz_partition_size = global_config.zero_optimization.get("zero_hpz_partition_size", None)

    # If not using MiCS, need to delete mics_shard_size and mics_hierarchical_params_gather
    # otherwise will get trigger assertion error when calling deepspeed.initialize
    if not global_config.zero_use_mics:
        zero_config = deepspeed_config.get("zero_optimization", None)
        if zero_config is not None:
            if "mics_shard_size" in zero_config:
                del zero_config["mics_shard_size"]
            if "mics_hierarchical_params_gather" in zero_config:
                del zero_config["mics_hierarchical_params_gather"]

    if hpz_partition_size is not None:
        groups._create_zero_param_parallel_group(hpz_partition_size)

    if global_config.zero_use_mics:
        with deepspeed.zero.MiCS_Init(
            config_dict_or_path=global_config.deepspeed_config,
            sequence_data_parallel_group=mpu.get_sequence_data_parallel_group(),
            dtype=global_config.params_dtype,
            enabled=use_zero_3,
            mpu=mpu,
        ):
            print_rank_0("Initializing Zero3 MiCS")
            model = get_model(global_config=global_config, use_cache=use_cache)
    else:
        with deepspeed.zero.Init(
            dtype=global_config.params_dtype,
            enabled=use_zero_3,
            zero_param_parallel_group=groups._ZERO_PARAM_INTRA_PARALLEL_GROUP,
            sequence_data_parallel_group=mpu.get_sequence_data_parallel_group(),
        ):
            print_rank_0("Initializing Zero3, hpZ partition size: {}".format(hpz_partition_size))
            model = get_model(global_config=global_config, use_cache=use_cache)

    if use_zero_3 and global_config.zero_use_leaf_modules:
        leaf_modules = global_config.zero_leaf_modules

        assert (
            leaf_modules is not None
        ), "leaf_modules must be specified when `use_leaf_modules` is set to True"

        from savanna.model import backbone, block
        from savanna.model.operators.attention import flash
        from savanna.model.operators.hyena import hyena
        from savanna.model.operators.local import base

        # check if leaf modules are available
        # TODO: jeromeku - make this more robust / flexible
        modules = []
        for m in leaf_modules:
            # Check and actually retrieve the modules from their attr names
            is_available = False
            valid_modules = [backbone, block, flash, hyena, base]
            for valid_module in valid_modules:
                if hasattr(valid_module, m):
                    modules.append(getattr(valid_module, m))
                    is_available = True
                    break

            assert (
                is_available
            ), f"leaf module {m} is not found in {valid_modules}, please double check or submit an issue"

        set_z3_leaf_modules(model, modules)
        print_rank_0(f"Marking following as leaf modules: {leaf_modules}")

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

    # jeromeku: We need to take into account the case where load is provided right at start of training
    # That is, we would like to provide the same save and load dir such that the same model config can be submitted 
    # to SLURM job again for job queuing and fault tolerance tools such as NVIDIA Heimdall.
    should_load = global_config.load is not None and (iteration > 0 or global_config.warmstart or global_config.finetune)
    
    if should_load:
        global_config.iteration = load_checkpoint(
            global_config=global_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iteration=iteration,
        )
        print_rank_0(f"SETUP_MODEL: Loaded checkpoint and starting from iteration {iteration}, warmstart={global_config.warmstart}")
    else:
        assert global_config.iteration == 0, "if not loading, must start from iteration 0"

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


def train_step(
    global_config,
    timers,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,
    noise_scale_logger,
    profiler: BaseProfiler,
):
    """Single training step."""

    if global_config.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            global_config=global_config,
            timers=timers,
            model=model,
            data_iterator=data_iterator,
            noise_scale_logger=noise_scale_logger,
            profiler=profiler,
        )
    else:
        losses = []

        for _ in range(global_config.gradient_accumulation_steps):
            # Forward model for one step.
            
            if straggler is not None:
                straggler_ctx = straggler.Detector.detection_section("TRAIN_STEP_FORWARD", profile_cuda=True)
                # print("Entering straggler.Detector context: TRAIN_STEP_FORWARD")
                straggler_ctx.__enter__()

            timers("forward").start()
            
            with profiler.mark("TRAIN_STEP_FORWARD"):
                outputs = forward_step(
                    global_config=global_config,
                    timers=timers,
                    data_iterator=data_iterator,
                    model=model,
                    is_train=True,
                    return_logits=False,
                )
            
            loss = outputs

            timers("forward").stop()

            losses.append(loss)

            if straggler is not None:
                straggler_ctx.__exit__(None, None, None)
                
            # Calculate gradients, reduce across processes, and clip.

            if straggler is not None:
                straggler_ctx = straggler.Detector.detection_section("TRAIN_STEP_BACKWARD", profile_cuda=True)
                # print("Entering straggler.Detector context: TRAIN_STEP_FORWARD")
                straggler_ctx.__enter__()
                
            timers("backward").start()
            with profiler.mark("TRAIN_STEP_BACKWARD"):
                backward_step(
                    global_config=global_config,
                    timers=timers,
                    optimizer=optimizer,
                    model=model,
                    loss=loss,
                )
            
            if straggler is not None:
                straggler_ctx.__exit__(None, None, None)

            if global_config.log_gradient_noise_scale:  # log noise scale if applicable
                noise_scale_logger.update()

            timers("backward").stop()


            if straggler is not None:
                straggler_ctx = straggler.Detector.detection_section("TRAIN_STEP_OPTIMIZER", profile_cuda=True)
                # print("Entering straggler.Detector context: TRAIN_STEP_FORWARD")
                straggler_ctx.__enter__()

            timers("optimizer").start()
            with profiler.mark("TRAIN_STEP_OPTIMIZER_STEP"):
                if global_config.deepspeed:
                    model.step()
                else:
                    raise ValueError("Must be using deepspeed to run savanna")
            timers("optimizer").stop()
            
            if straggler is not None:
                straggler_ctx.__exit__(None, None, None)

        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean(),
        }  # reduces losses across machines for logging

    if global_config.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


def train_step_pipe(global_config, timers, model, data_iterator, noise_scale_logger, profiler: BaseProfiler):
    """Single training step with DeepSpeed's pipeline parallel engine."""
    assert global_config.deepspeed

    with profiler.mark("TRAIN_STEP_PIPE"):
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

    # MP: safe_get_full_grad will not work with pipe parallel and zero 1
    if noise_scale_logger is not None:
        noise_scale_logger.update()

    return loss_dict


def train(
    global_config,
    timers,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
    valid_data_iterator_list=None,
):
    """Train the model function."""
    if global_config.recycle_events:
        print(f"rank{torch.distributed.get_rank()}: Recycling events")
        monkey_patch_event()
    
    # if global_config.patch_record_stream:
    #     assert (
    #         os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS", "0") == "1"
    #     ), "Must set TORCH_NCCL_AVOID_RECORD_STREAMS=1 when using record_stream_monkey_patch()"
    #     if torch.distributed.get_rank() == 0:
    #         print("Monkey patching torch record streams")
    #     # record_stream_monkey_patch(flush_frequency=global_config.record_stream_flush_frequency, backlog=global_config.record_stream_backlog)

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
    noise_scale_logger = get_noise_scale_logger(global_config, model)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)

    # Initialize torch.profiler
    # if profiling is not enabled, will return a null context / no-op profiler
    # NOTE: if profiling is not enabled, will return a empty profiler
    # any calls to the profiler will be no-ops subsequently
    # All profiler types inherit from BaseProfiler
    profiler: BaseProfiler = setup_profiler(global_config)

    alloc_stats = AllocStats(torch.distributed.get_rank(), global_config=global_config)

    profiler.start()

    #Heimdall Straggler Detector
    total_flops = 0.0
    
    # For debugging comms ops

    comms_logging_enabled = global_config.deepspeed_enable_comms_logging
    ds_comms_logging_interval = global_config.deepspeed_comms_logging_interval
    if ds_comms_logging_interval is None:
        ds_comms_logging_interval = global_config.log_interval

    ds_comms_logger = ds_comms.comms_logger

    prof_ops = global_config.deepspeed_comms_ops_to_log

    if prof_ops is None:
        prof_all = True
    else:
        prof_all = False

    world_size = torch.distributed.get_world_size()
    rank_logs = list(range(0, world_size, global_config.deepspeed_comms_ranks_to_log))

    configure_deepspeed_comms_logging(
        enable=comms_logging_enabled,
        verbose=global_config.deepspeed_comms_logging_verbose,
        debug=global_config.deepspeed_comms_logging_debug,
        prof_all=prof_all,
        prof_ops=prof_ops,
        rank_logs=rank_logs,
    )
    if torch.distributed.get_rank() == 0 and comms_logging_enabled:
        print(f"Comms logger enabled: {ds_comms_logger.enabled=}")
        print(f"Comms verbose: {ds_comms_logger.verbose=}")
        print(f"Comms debug: {ds_comms_logger.debug=}")
        print(f"Comms prof_all: {ds_comms_logger.prof_all=}")
        print(f"Comms prof_ops: {ds_comms_logger.prof_ops=}")
        print(f"COMMS LOG RANKS: {rank_logs=}")

    if global_config.disable_gc:
        print(f"rank{torch.distributed.get_rank()}: Disabling automatic gc, collection generation {global_config.gc_collect_generation}")
        gc.disable()
        
    while iteration < global_config.train_iters:
        
        if global_config.disable_gc:
            gc.collect(generation=global_config.gc_collect_generation)
        
        profiler.step()
        alloc_stats.step(iteration)

        if global_config.async_save:
            finalize_async_save(global_config, blocking=False)
    
        if comms_logging_enabled and iteration % ds_comms_logging_interval == 0:
            enable_deepspeed_comms_logging()
            if torch.distributed.get_rank() == 0:
                print(f"Comms logging enabled at iteration {iteration}")

        loss_dict, skipped_iter = train_step(
            global_config=global_config,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            noise_scale_logger=noise_scale_logger,
            profiler=profiler,
        )
        iteration += 1
        global_config.iteration = iteration

        total_flops += get_total_flops(global_config, model)
        
        if global_config.train_data_token_index is not None:
            global_config.train_data_token_index += global_config.train_batch_size * global_config.seq_length

        if global_config.precision == "fp16":
            overflow_monitor.check(skipped_iter)  # check for repeated overflow

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # @jeromeku: per-rank mem alloc counts
        # meant to be used with per-rank logging
        should_print_mem_stats = (
            global_config.print_mem_alloc_stats
            and torch.distributed.get_rank() % global_config.mem_alloc_stats_ranks == 0
        )

        if should_print_mem_stats:
            print_mem_alloc_stats(iteration)

        if comms_logging_enabled and iteration % ds_comms_logging_interval == 0:
            if global_config.deepspeed_comms_print_summary:
                ds_comms_logger.log_all(print_log=True if torch.distributed.get_rank() == 0 else False)
            serialize_comms_dict()
            disable_deepspeed_comms_logging()

        if straggler is not None:
            if global_config.use_next and iteration % global_config.next_report_interval == 0:
                rank = torch.distributed.get_rank()
                # print_local_summaries(report, rank)
                report = straggler.Detector.generate_report()
                if report is not None:
                    print(f"rank{rank}: Generated straggler report")
                    print_straggler_report(report, 
                                           rank=rank, 
                                           iteration=iteration, 
                                           gpu_scores=global_config.next_gpu_scores,
                                           gpu_rel_threshold=global_config.next_gpu_rel_threshold,
                                           gpu_individual_threshold=global_config.next_gpu_individual_threshold,
                                           section_scores=global_config.next_section_scores, 
                                           section_rel_threshold=global_config.next_section_rel_threshold,
                                           section_individual_threshold=global_config.next_section_individual_threshold,
                                           stragglers=global_config.next_stragglers)
                    # print(report)


        # Heimdall Straggler Detector
        if global_config.heimdall_log_straggler and iteration % global_config.heimdall_log_interval == 0:
            hsd_timer.report(total_flops, global_config.heimdall_log_interval)
            # print(f"rank{torch.distributed.get_rank()}: Generating hsd report, Total FLOPS: {total_flops} has report: {has_report}")
            total_flops = 0.0
            
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
        if (
            global_config.eval_per_ds_interval
            and iteration % global_config.eval_per_ds_interval == 0
            and global_config.do_per_ds_valid
            and valid_data_iterator_list is not None
        ):
            evaluate_multiple_datasets_and_print_results(
                global_config=global_config,
                forward_step_func=forward_step,
                valid_data_iterator_list=valid_data_iterator_list,
                model=model,
                iteration=iteration,
                timers=timers,
                )

        # if global_config.exit_interval and iteration % global_config.exit_interval == 0:
        #     torch.distributed.barrier()
        #     time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     rank = torch.distributed.get_rank()
        #     print_rank_0(
        #         "rank: {} | time: {} | exiting the program at iteration {}".format(rank, time_str, iteration)
        #     )
        #     sys.exit()

    return iteration


def evaluate(global_config, forward_step_fn, data_iterator, model, verbose=False, timers=None, evaluating_per_ds=False):
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
    if evaluating_per_ds:
        eval_iters = global_config.eval_per_ds_iters
    else:
        eval_iters = global_config.eval_iters

    with torch.no_grad():
        iteration = 0
        while iteration < eval_iters:
            iteration += 1
            if verbose and iteration % global_config.log_interval == 0:
                print_rank_0("Evaluating iter {}/{}".format(iteration, eval_iters))

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
        evaluating_per_ds=False,
    )

    print_and_log_results(total_loss_dict, chart_name, prefix, iteration, global_config)


def evaluate_multiple_datasets_and_print_results(
    global_config,
    forward_step_func,
    valid_data_iterator_list,
    model,
    iteration,
    timers=None,
):
    """Evaluate multiple datasets and aggregate their results."""

    # Dictionary to store aggregated losses by dataset name
    dataset_losses = {}
    dataset_counts = {}
    rank = torch.distributed.get_rank()
    # Evaluate each dataset and aggregate results
    for i, valid_data_iterator in enumerate(valid_data_iterator_list):
        
        dataset_name = global_config.per_ds_valid_data_paths[i]
        dataset_name = "/".join(dataset_name.split("/")[3:]).replace("_text_CharLevelTokenizer_document","")
        
        prefix = f"iteration {iteration}"
        if rank == 0:
            print(f"PER_DS_EVALUATION: Validating on dataset {i}: {dataset_name}", flush=True)
        
        # Get losses for this dataset
        total_loss_dict = evaluate(
            global_config=global_config,
            forward_step_fn=forward_step_func,
            data_iterator=valid_data_iterator,
            model=model,
            verbose=False,
            timers=timers,
            evaluating_per_ds=True,
        )

        # Initialize or update aggregated losses for this dataset name
        if dataset_name not in dataset_losses:
            dataset_losses[dataset_name] = total_loss_dict
            dataset_counts[dataset_name] = 1
        else:
            # Add losses
            for key, value in total_loss_dict.items():
                if isinstance(value, dict):
                    if key not in dataset_losses[dataset_name]:
                        dataset_losses[dataset_name][key] = value
                    else:
                        for k2, v2 in value.items():
                            dataset_losses[dataset_name][key][k2] += v2
                else:
                    if key not in dataset_losses[dataset_name]:
                        dataset_losses[dataset_name][key] = value
                    else:
                        dataset_losses[dataset_name][key] += value
            dataset_counts[dataset_name] += 1

    # Calculate averages and log results
    for dataset_name, losses in dataset_losses.items():
        count = dataset_counts[dataset_name]
        
        # Average each metric
        averaged_losses = {}
        for k, v in losses.items():
            if isinstance(v, dict):
                averaged_losses[k] = {}
                for k2, v2 in v.items():
                    averaged_losses[k][k2] = v2 / count
            else:
                averaged_losses[k] = v / count

        # Use helper function to print and log results
        print_and_log_results(
            averaged_losses,
            f"validation/{dataset_name}",
            prefix,
            iteration,
            global_config
        )


def print_and_log_results(total_loss_dict, chart_name, prefix, iteration, global_config):
    """Helper function to format, print and log evaluation results."""
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
    print_rank_0("-" * length)
    print_rank_0("-" * length)
    print_rank_0("-" * length)
    print_rank_0("-" * length)
    print_rank_0("-" * length)
