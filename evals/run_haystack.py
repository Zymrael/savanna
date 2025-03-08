import os
import time
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml

from evals.data import prepare_inputs
from evals.retrieval import (
    NeedleInAHaystackArgs,
    check_indices,
    generate_power_of_two_range,
    generate_random_dna_sequence,
    get_contacts,
    repeat_sequence_until,
)
from evals.savanna_utils import get_model, get_timestamp, init_savanna, load_checkpoint
from savanna import mpu, print_rank_0
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
        print_rank_0(f"run_forward after context parallel gather: {logits.shape}", debug=True)
    
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
    
def run_batches(model, input_ids, vocab_ids, global_config, batch_size=1, padding_size=0):
    outs = []
    num_batches = input_ids.shape[0]
    
    for i in range(0, num_batches, batch_size):
        batch = input_ids[i:i+batch_size]
        out = run_one_batch(model, batch, vocab_ids, global_config, padding_size=padding_size)
        outs.append(out)
    return np.concatenate(outs, axis=0)

def get_categorical_jacobian(
    seq: str,
    model,
    tokenizer,
    effective_vocab: List[str] = list("ACGT"),
    batch_size: int = None,
    perturb_indices: List[int] = None,
    measure_indices: List[int] = None,
    device: str = "cuda",
    verbose: bool = False,
) -> np.ndarray:
    """
    Computes the categorical Jacobian tensor with shape (L, V, L, V), where L
    is the sequence length and V is the effective vocabulary size. This
    effective vocabulary is specified in the `effective_vocab` parameter.
    The tensor is compute for a given sequence and `StripedHyena` model.

    Supports partial Jacobian computation. `perturb_indices` restricts the
    positions in the sequence that are mutated, and `measure_indices` restricts
    the logit values that are stored.
    """
    rank = torch.distributed.get_rank()
    L = len(seq)
    V = len(effective_vocab)
    
    if batch_size is None:
        batch_size = V
    
    if perturb_indices is None:
        perturb_indices = list(range(L))
    else:
        check_indices(perturb_indices, L)
    if measure_indices is None:
        measure_indices = list(range(L))
    else:
        check_indices(perturb_indices, L)
    
    L_perturb = len(perturb_indices)
    L_measure = len(measure_indices)
    
    if verbose:
        print_rank_0(f"L_perturb: {L_perturb}, L_measure: {L_measure}", debug=True)

    compute_partial = L_perturb != L and L_measure != L

    # All mp shards see the same seq length, which is original seq length / cp_size / mp_size
    input_ids, padding_size = prepare_inputs(
        seq=seq,
        global_config=global_config,
        tokenizer=tokenizer,
        device="cuda"
    )

    if verbose:
        print_rank_0(f"inputs: {input_ids.shape}", debug=True)
        print_rank_0(f"padding_size: {padding_size}", debug=True)

    vocab_ids = torch.tensor(
        tokenizer.tokenize("".join(effective_vocab)),
        dtype=torch.int,
    ).to(device)

    # Compute the unperturbed logits.
    fx = run_batches(model, input_ids, vocab_ids, global_config, batch_size, padding_size=padding_size)

    if verbose:
        print_rank_0(f"fx shape after run_batches: {fx.shape}", debug=True)
    
    if L_measure == L:
        fx = fx[0]
    else:
        fx = fx[0, measure_indices]
    
    if verbose:
        print_rank_0(f"fx shape after selecting measure indices: {fx.shape}", debug=True)

    inputs_tiled = torch.tile(input_ids, [V, 1])
    
    if verbose:
        print_rank_0(f"inputs_tiled shape: {inputs_tiled.shape}", debug=True)
    
    fx_h = np.zeros((L_perturb, V, L_measure, V))
  
    if verbose:
        print_rank_0(f"fxh shape: {fx_h.shape}", debug=True)

    if verbose:
        print_rank_0(f"perturb_indices: {perturb_indices[:10]}", debug=True)
    
    print_rank_0(f"{get_timestamp()}: Running pertubations with L: {L}, L_measure: {L_measure}, perturb_indices: {len(perturb_indices)}, measure_indices: {len(measure_indices)}", debug=True)

    for i, perturb_index in enumerate(perturb_indices):
        if i % 10 == 0 and rank == 0:
            print_rank_0(f"{get_timestamp()}: Running pertubation {i} {perturb_index}", debug=True)
        
        x_h = torch.clone(inputs_tiled).to(device)
        x_h[:, perturb_index] = vocab_ids
        
        out = run_batches(model, x_h, vocab_ids, global_config, batch_size, padding_size=padding_size)
        
        if verbose:
            print_rank_0(f"perturb_index {perturb_index} out shape: {out.shape}", debug=True)
 
        if L_measure == L:
            out = out
        else:
            out = out[:, measure_indices]
        
        fx_h[i] = out

       
    # Compute the difference between the perturbed and unperturbed logits.
    jac = fx_h - fx  # Broadcasts the unperturbed matrix.

    assert len(jac.shape) == 4, "Jacobian must have 4 dimensions."

    for i in range(4):  # Center values across each dimension.
        jac -= jac.mean(i, keepdims=True)

    if verbose and compute_partial:
        print_rank_0(f"Partial Jacobian computation currently does not support symmetrization, results will be different from full Jacobian computation.", debug=True)
        
    if not compute_partial:
        jac = (jac + jac.transpose(2, 3, 0, 1)) / 2.0

    return jac

def run_needle_in_a_haystack(global_config, model, tokenizer):
    verbose = False
    rank = torch.distributed.get_rank()
    
    args = NeedleInAHaystackArgs.from_global_config(global_config)
    print_rank_0(f"NeedleInAHaystackArgs: {args}", debug=True)
    
    needle_seq = generate_random_dna_sequence(args.needle_length)
    print_rank_0(f"Inserting needle: {needle_seq}", debug=True)

    haystack_lengths = generate_power_of_two_range(
        args.haystack_min_length,
        args.haystack_max_length,
    )

    data = []

    with open(args.background_sequence_path, "r") as f:
        original_background_seq = f.read()

    print_rank_0(f"Background sequence length: {len(original_background_seq)}", debug=True)
    
    if rank == 0:
        start = time.time()

    output_dir = os.path.join(args.output_dir, f"{args.model_name}", f"{args.haystack_min_length}-{args.haystack_max_length}")
    if rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yml"), "w") as f:
            yaml.dump(asdict(global_config), f)
    torch.distributed.barrier()
  
    for haystack_length in haystack_lengths:
        background_seq = repeat_sequence_until(original_background_seq, haystack_length)
        
        if rank == 0:
            division_start = time.time()
        
        for division in range(args.n_divisions):
            print_rank_0(f"{get_timestamp()}: Haystack length {haystack_length}, division {division}", debug=True)

            insert_idx_start = division * (haystack_length // args.n_divisions)
            insert_idx_end = insert_idx_start + args.needle_length
            if insert_idx_end > len(background_seq):
                print_rank_0("Needle is longer than a division of the context. Skipping.", debug=True)
                # Needle is longer than a division of the context.
                continue
        
            print_rank_0(f"Inserting needle at {insert_idx_start} to {insert_idx_end}", debug=True)
            context = background_seq[:insert_idx_start] + needle_seq + background_seq[insert_idx_end:]
            seq = context + needle_seq

            perturb_start = max(0, insert_idx_start - args.perturb_flank)
            perturb_end = min(len(seq), insert_idx_end + args.perturb_flank)

            print_rank_0(f"{get_timestamp()}: Perturb indices: {perturb_start}:{perturb_end}, Measure indices: {len(context)}:{len(seq)}, Sequence length {len(seq)}", debug=True)
        
            jac = get_categorical_jacobian(
                seq,
                model,
                tokenizer,
                effective_vocab="ACGT",
                perturb_indices=list(range(perturb_start, perturb_end)),
                measure_indices=list(range(len(context), len(seq))),
                batch_size=args.haystack_batch_size,
                device="cuda",
                verbose=verbose
            )
            contacts = get_contacts(jac, rm=0)

            score = (
                np.mean(
                    [contacts[insert_idx_start - perturb_start + i, i] for i in range(args.needle_length)]
                )
                / 2.0
            )
           
            if rank == 0:
                division_end = time.time()
                print(f"{get_timestamp()}: Division {division} time: {division_end - division_start} seconds", flush=True)
                print(f"{get_timestamp()}: Haystack length {haystack_length}, division {division}, score: {score}", flush=True)

            data.append(
                [
                    haystack_length,
                    int(round((1.0 - (division / args.n_divisions)) * 100)),
                    score,
                ]
            )

        torch.distributed.barrier()

        if rank == 0:
            df = pd.DataFrame(
                data,
                columns=[
                    "haystack_length",
                    "depth",
                    "score",
                ],
            )
            df.to_csv(os.path.join(output_dir, f"{haystack_length}_{args.needle_length}.csv"), index=False)

        torch.distributed.barrier()
        print_rank_0(f"{get_timestamp()}: Completed haystack length {haystack_length}", debug=True)
    
    torch.distributed.barrier()
    print_rank_0(f"{get_timestamp()}: Finished running needle in a haystack, outputs in {output_dir}")
    
    if rank == 0:
        end = time.time()
        print_rank_0(f"{get_timestamp()}: Time taken: {end - start} seconds", debug=True)

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

    # Run forward
    run_needle_in_a_haystack(global_config, model, tokenizer)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
