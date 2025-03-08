import os
import time
from dataclasses import asdict, dataclass, fields
from typing import List

import pickle
import numpy as np
import pandas as pd
import torch
import yaml

from evals.data import prepare_inputs
from evals.retrieval import (
    check_indices,
    generate_power_of_two_range,
    generate_random_dna_sequence,
    get_contacts,
    repeat_sequence_until,
)
from evals.savanna_utils import get_model, get_timestamp, init_savanna, load_checkpoint
from savanna import mpu, print_rank_0
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron


def run_forward(model, input_ids, global_config, padding_size=0):
    
    if mpu.get_sequence_parallel_world_size() > 1:
        original_input_ids = input_ids

        input_ids = mpu.zigzag_split_across_cp_ranks(input_ids)
        # Check zig zag reconstruction
        reconstructed_input_ids = mpu.zigzag_gather_from_cp_ranks(input_ids)
        #print_rank_0(f"run_forward {reconstructed_input_ids[:, -padding_size:]=}", debug=True)
        assert torch.allclose(
            original_input_ids, reconstructed_input_ids
        ), f"Zigzag reconstruction failed, {original_input_ids.shape=} != {reconstructed_input_ids.shape=}"
        #print_rank_0(f"run_forward after zigzag split {input_ids.shape=}", debug=True)

    inputs = (input_ids, None, None) # savanna expects a tuple of (input_ids, position_ids, attention_mask)
    logits = model(inputs, global_config=global_config)
   
    if global_config.model_parallel_size > 1:
        logits = mpu.gather_from_model_parallel_region(logits)
        #print_rank_0(f"run_forward after model parallel gather: {logits.shape}", debug=True)
    
    if global_config.context_parallel_size > 1:
        logits = mpu.zigzag_gather_from_cp_ranks(logits)
        # print_rank_0(f"run_forward after context parallel gather: {logits.shape}", debug=True)
    
    #print_rank_0(f"run_forward before removing padding: {logits.shape}", debug=True)
    # Remove padding before gathering along seq dim
    if padding_size > 0:
        logits = logits[:, :-padding_size, :]
    #print_rank_0(f"run_forward after removing padding: {logits.shape}", debug=True)

    
    return logits

def run_one_batch(model, input_ids, vocab_ids, global_config, padding_size=0):
    with torch.inference_mode():
        logits = run_forward(model, input_ids, global_config, padding_size=padding_size)
        #print_rank_0(f"logits shape after run_forward: {logits.shape}", debug=True)
        
        logits = logits[..., vocab_ids].float().cpu().numpy()
        #print_rank_0(f"logits shape after vocab_ids: {logits.shape}", debug=True)
        return logits

def run_forward_passes(global_config, model, tokenizer):
    verbose = False
    # sequences = ["ATGCATGCATGCATGCATGCATGC", "TATATATATATATATATATATATATATATA", "ATATATA"]
    sequences = "ATGCATGCATGCATGCATGCATGC"

    def byte(seqlist):
        return [seq.encode("utf-8") for seq in seqlist]
    # sequences = byte(sequences)

    rank = torch.distributed.get_rank()
    
    input_path = global_config.input_path
    # os.environ['FWD_INPUT_PATH']
    output_path = global_config.output_path
    # os.environ['FWD_OUTPUT_PATH']

    if input_path:
        with open(input_path, "r") as f:
            sequences = f.readlines()
            sequences = [seq.strip() for seq in sequences]

    data = []
            
    torch.distributed.barrier()

    device = "cuda"
    vocab_ids = torch.tensor(range(512)).to(device)

    outs = []

    num_batches = len(sequences)
    
    for i in range(0, num_batches):
        seq = sequences[i]

        input_ids, padding_size = prepare_inputs(
            seq=seq,
            global_config=global_config,
            tokenizer=tokenizer,
            device="cuda"
        ) ### TODO set up batching inside here, maintain compatibility with haystack

        batch = input_ids
        out = run_one_batch(model, batch, vocab_ids, global_config, padding_size=padding_size)
        outs.append(out)

    torch.distributed.barrier()

    if rank == 0:
        # TODO h5py for efficiency
        with open(output_path, 'wb') as f:
            pickle.dump(outs, f)

    torch.distributed.barrier()
    print_rank_0(f"{get_timestamp()}: Completed forward passes {len(sequences)}", debug=True)

    torch.distributed.barrier()


if __name__ == "__main__":

    global_config = init_savanna()

    # Tokenizer
    global_config.build_tokenizer()
    tokenizer = global_config.tokenizer

    # Init distributed and setup mpu
    initialize_megatron(global_config=global_config)

    rank = torch.distributed.get_rank()
    
    # Need to cast model to dtype -- this is usually done by deepspeed.initialize
    model = get_model(global_config, cast_to_dtype=True)
    
    if global_config.load:
        model = load_checkpoint(model, global_config)

    torch.distributed.barrier()

    run_forward_passes(global_config, model, tokenizer)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
