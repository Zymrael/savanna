"""Train"""

import os
from datetime import datetime
from typing import List, Tuple

import deepspeed
import torch

from savanna import mpu, print_rank_0
from savanna.arguments import GlobalConfig
from savanna.checkpointing import read_global_step
from savanna.initialize import initialize_megatron
from savanna.memory_stats import maybe_enable_memory_snapshot
from savanna.model import BackbonePipe
from savanna.model_loading import get_all_shard_files, get_shard_by_mp_rank
from savanna.tokenizer import CharLevelTokenizer
from savanna.utils import (
    FP8_SHAPE,
    get_ltor_masks_and_position_ids,
    make_upper_case,
    pad_to_multiple,
)


def get_model(global_config, use_cache=False):
    # """Build the model."""

    # # Temporarily disable mup so that the base model does not use the mup init functions before set_base_shapes is called below.
    # # If mup isn't being used anyways, this has no effect.
    # old_use_mup = global_config.use_mup
    # global_config.use_mup = False
    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    model = model.to_sequential()

    #    global_config.use_mup = old_use_mup
    return model


def check_te_env_vars(global_config):
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

    # These should be exported by the launcher script
    if global_config.use_cp_flash_te:
        assert (
            os.environ["NVTE_FLASH_ATTN"] == "1"
        ), "NVTE_FLASH_ATTN must be enabled when use_cp_flash_te = True"
        assert (
            os.environ["NVTE_FUSED_ATTN"] == "0"
        ), "NVTE_FUSED_ATTN must be disabled when use_cp_flash_te = True"
        assert (
            os.environ["NVTE_UNFUSED_ATTN"] == "0"
        ), "NVTE_UNFUSED_ATTN must be disabled when use_cp_flash_te = True"

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


def prepare_batch(
    global_config: GlobalConfig,
    seqs: List[str],
    device: torch.device,
    tokenizer: CharLevelTokenizer,
    prepend_bos: bool = False,
) -> Tuple[torch.Tensor, List[int]]:

    seq_lengths = [len(seq) for seq in seqs]
    max_seq_length = max(seq_lengths)

    should_pad_to_fp8 = any(
        getattr(global_config, attr) for attr in dir(global_config) if attr.startswith("use_fp8")
    )
    print_rank_0(f"DEBUG::INFERENCE::prepare_batch::should_pad_to_fp8 {should_pad_to_fp8}")
    if should_pad_to_fp8:
        # NOTE: @jeromeku: seq length to mutliple of 8 in order to use TE FP8
        max_seq_length_padded = pad_to_multiple(max_seq_length, FP8_SHAPE[0])
        max_seq_length = max_seq_length_padded
        assert (
            max_seq_length % FP8_SHAPE[0] == 0
        ), f"max_seq_length {max_seq_length} must be a multiple of {FP8_SHAPE[0]}"

    print_rank_0(f"DEBUG::INFERENCE::prepare_batch::max_seq_length {max_seq_length}")
    input_ids = []
    for seq in seqs:
        padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
        input_ids.append(
            torch.tensor(
                ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
                dtype=torch.long,
            )
            .to(device)
            .unsqueeze(0)
        )
    input_ids = torch.cat(input_ids, dim=0)
    print_rank_0(f"DEBUG::INFERENCE::prepare_batch::input_ids {input_ids.shape=}")
    input_ids, _ = make_upper_case(input_ids) # This was used during pretraining / extension
    pad_token = global_config.tokenizer.pad

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=input_ids,
        eod_token=global_config.tokenizer.eod,
        pad_token=pad_token,
        eod_mask_loss=global_config.eod_mask_loss,
        pad_mask_loss=global_config.pad_mask_loss,
        materialize_attn_mask=global_config.materialize_attn_mask,
    )

    # zigzag split across cp ranks
    if mpu.get_sequence_parallel_world_size() > 1:
        original_input_ids = input_ids

        input_ids = mpu.zigzag_split_across_cp_ranks(input_ids)
        attention_mask = mpu.zigzag_split_across_cp_ranks(attention_mask, -2) if global_config.materialize_attn_mask else None
        position_ids = mpu.zigzag_split_across_cp_ranks(position_ids)

        # Check zig zag reconstruction
        reconstructed_input_ids = mpu.zigzag_gather_from_cp_ranks(input_ids)
        assert torch.allclose(original_input_ids, reconstructed_input_ids), f"Zigzag reconstruction failed, {original_input_ids.shape=} != {reconstructed_input_ids.shape=}"
    
    return (
        input_ids,
        attention_mask if global_config.materialize_attn_mask else None,
        position_ids,
    )


def generate_seqs(batch_size=1, seq_length=8192, vocab="GATC"):
    from numpy.random import randint

    # Generate random batch size sequences of length seq_length
    seqs = []
    for _ in range(batch_size):
        seqs.append("".join([vocab[i] for i in randint(0, len(vocab), seq_length)]))

    return seqs


def print_params(model, msg=""):
    for mname, module in model.named_modules():
        for pname, param in module.named_parameters():
            print_rank_0(f"DEBUG::INFERENCE::params::{msg}::{pname} {param.shape} {param.dtype}")

def print_buffers(model, msg=""):
    for mname, module in model.named_modules():
        for bname, buffer in module.named_buffers():
            print_rank_0(f"DEBUG::INFERENCE::buffers::{msg}::{bname} {buffer.shape} {buffer.dtype}")

if __name__ == "__main__":

    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()
    torch.set_printoptions(precision=5)

    ## Tokenizer
    global_config.build_tokenizer()
    tokenizer = global_config.tokenizer

    # Init distributed and setup mpu
    initialize_megatron(global_config=global_config)

    rank = torch.distributed.get_rank()
    seq_length = global_config.seq_length
    seqs = generate_seqs(batch_size=1, seq_length=seq_length)
    device = torch.cuda.current_device()

    inputs = (input_ids, attention_mask, position_ids) = prepare_batch(
        global_config=global_config, seqs=seqs, tokenizer=tokenizer, device=device
    )

    seq_length = input_ids.shape[1]
    is_context_parallel = global_config.context_parallel_size > 1
    seq_length = seq_length * mpu.get_sequence_parallel_world_size() if is_context_parallel else seq_length
    print_rank_0(f"DEBUG::INFERENCE::after prepare_batch::seq_length {seq_length}")
    # Need to adjust seq_length for the model
    global_config.seq_length = seq_length
    vocab_size = global_config.padded_vocab_size
    dtype = global_config.params_dtype

    #Check cp zig zag
    reconstructed_input_ids = mpu.zigzag_gather_from_cp_ranks(input_ids)
    print(f"DEBUG::INFERENCE::reconstructed_input_ids {reconstructed_input_ids.shape=}")

    # Need to cast model to dtype -- this is usually done by deepspeed.initialize
    model = get_model(global_config).to(dtype)
    
    print_rank_0(f"DEBUG::INFERENCE AFTER MODEL_INIT::")
    print_params(model, "AFTER MODEL_INIT")
    print_buffers(model, "AFTER MODEL_INIT")

    if global_config.load:
        # Need to account for context parallel
        # Shard order is TP then CP
        load_rank = rank % global_config.model_parallel_size
        world_size = torch.distributed.get_world_size()
        assert world_size == global_config.model_parallel_size * global_config.context_parallel_size, f"Only tp and cp are supported, {world_size=} != {global_config.model_parallel_size=} * {global_config.context_parallel_size=}"

        tag = read_global_step(global_config.load)
        checkpoint_dir = os.path.join(global_config.load, f"global_step{tag}")
        checkpoint_files = get_all_shard_files(checkpoint_dir)
        print(f"DEBUG::LOAD_CHECKPOINT rank {rank} load_rank {load_rank}")
        shard_file = get_shard_by_mp_rank(checkpoint_dir, load_rank)

        # NOTE: @jeromeku: this is not needed since device is set in `_init_distributed`
        # world_size = torch.distributed.get_world_size()
        # local_world_size = torch.cuda.device_count()
        # local_rank = rank % local_world_size
        # torch.cuda.set_device(local_rank)
        # device = torch.cuda.current_device()

        state_dict = torch.load(shard_file, map_location="cpu")["module"]
        model.load_state_dict(state_dict)
        model.eval()
        print_params(model, "AFTER LOAD_STATE_DICT")
        print_buffers(model, "AFTER LOAD_STATE_DICT")

        # NOTE: @jeromeku: this is not needed since model is already set in BackbonePipe.__super__ (PipelineModule)
        # model = model.to(device)

    elif global_config.save:
        model, *_ = deepspeed.initialize(model=model, args=global_config, dist_init_required=False, mpu=mpu)
        print_params(model, "AFTER DEEPSPEED_INIT")
        print_buffers(model, "AFTER DEEPSPEED_INIT")
        model.save_checkpoint(global_config.save)  # model.save_checkpoint(global_config.save)

    torch.distributed.barrier()
    try:
        with maybe_enable_memory_snapshot(global_config, global_step=0) as memory_profiler:
            logits = model(inputs, global_config=global_config)
            logits = mpu.gather_from_model_parallel_region(logits)
            logits_gathered = mpu.zigzag_gather_from_cp_ranks(logits)

            if memory_profiler is not None:
                memory_profiler.step()
        
        if global_config.save:
            save_path = os.path.join(global_config.save, f"global_step0/logits/ref/rank{rank}_logits.pt")
            gathered_save_path = os.path.join(global_config.save, f"global_step0/logits/ref/rank{rank}_logits_gathered.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        elif global_config.load:
            save_path = os.path.join(global_config.load, f"global_step0/logits/test/rank{rank}_logits.pt")
            gathered_save_path = os.path.join(global_config.load, f"global_step0/logits/test/rank{rank}_logits_gathered.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            raise ValueError("Either save or load must be set")

        torch.save(logits, save_path)
        print_rank_0(f"DEBUG::INFERENCE::{rank=}::logits {logits.shape=} saved to {save_path}") 
        torch.save(logits_gathered, gathered_save_path)
        print_rank_0(f"DEBUG::INFERENCE::{rank=}::logits_gathered {logits_gathered.shape=} saved to {gathered_save_path}")
    except Exception as e:
        print_rank_0(f"DEBUG::INFERENCE::{rank=} Exception {e}")
        raise e
    finally:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()  # torch.distributed.destroy_process_group()    torch.distributed.destroy_process_group()    # torch.distributed.destroy_process_group()
