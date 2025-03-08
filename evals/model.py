
import torch

from evals.data import generate_seqs, prepare_batch
from savanna import mpu, print_rank_0
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.model import BackbonePipe
from savanna.model_loading import get_shard_by_mp_rank


def get_model(global_config, use_cache=False, cast_to_dtype=True):
    model = BackbonePipe(
        global_config=global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    model = model.to_sequential()

    return model.to(global_config.params_dtype) if cast_to_dtype else model


def print_params(model, msg=""):
    for mname, module in model.named_modules():
        for pname, param in module.named_parameters():
            print_rank_0(f"DEBUG::INFERENCE::params::{msg}::{pname} {param.shape} {param.dtype}")

def print_buffers(model, msg=""):
    for mname, module in model.named_modules():
        for bname, buffer in module.named_buffers():
            print_rank_0(f"DEBUG::INFERENCE::buffers::{msg}::{bname} {buffer.shape} {buffer.dtype}")


def init_savanna():
    global_config = GlobalConfig.consume_global_config()
    global_config.configure_distributed_args()
    return global_config

def load_checkpoint(model, global_config):
    # NOTE: @jeromeku: this is not needed since device is set in `_init_distributed`
    # world_size = torch.distributed.get_world_size()
    # local_world_size = torch.cuda.device_count()
    # local_rank = rank % local_world_size
    # torch.cuda.set_device(local_rank)
    # device = torch.cuda.current_device()

    load_rank = rank % global_config.model_parallel_size
    world_size = torch.distributed.get_world_size()
    assert world_size == global_config.model_parallel_size * global_config.context_parallel_size, f"Only tp and cp are supported, {world_size=} != {global_config.model_parallel_size=} * {global_config.context_parallel_size=}"

    #checkpoint_files = get_all_shard_files(global_config.load)
    print(f"DEBUG::LOAD_CHECKPOINT rank {rank} load_rank {load_rank}")
    checkpoint_dir = global_config.load
    shard_file = get_shard_by_mp_rank(checkpoint_dir, load_rank)

    state_dict = torch.load(shard_file, map_location="cpu")["module"]
    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_forward(model, inputs, global_config):
    logits = model(inputs, global_config=global_config)
 
    if global_config.model_parallel_size > 1:
        logits = mpu.gather_from_model_parallel_region(logits)
    if global_config.context_parallel_size > 1:
        logits = mpu.zigzag_gather_from_cp_ranks(logits)
 
    return logits

    
if __name__ == "__main__":

    global_config = init_savanna()

    # Tokenizer
    global_config.build_tokenizer()
    tokenizer = global_config.tokenizer

    # Init distributed and setup mpu
    initialize_megatron(global_config=global_config)

    rank = torch.distributed.get_rank()
    seq_length = global_config.seq_length
    seqs = generate_seqs(batch_size=1, seq_length=seq_length)
    device = torch.cuda.current_device()

    # inputs = (input_ids, attention_mask, position_ids)
    inputs, global_config = prepare_batch(
        global_config=global_config, seqs=seqs, tokenizer=tokenizer, device=device
    )
    
    # Need to cast model to dtype -- this is usually done by deepspeed.initialize
    model = get_model(global_config, cast_to_dtype=True)
    
    if global_config.load:
        model = load_checkpoint(model, global_config)

    torch.distributed.barrier()

    # Run forward
    logits = run_forward(model, inputs, global_config)

    if global_config.save_logits:
        if rank == 0:
            torch.save(logits, global_config.save_logits)
            print_rank_0(f"DEBUG::INFERENCE::{rank=}::logits {logits.shape=} saved to {global_config.save_logits}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()  # torch.distributed.destroy_process_group()    torch.distributed.destroy_process_group()    # torch.distributed.destroy_process_group()
