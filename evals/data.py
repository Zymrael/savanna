from typing import List, Tuple

import torch

from savanna import mpu, print_rank_0
from savanna.arguments import GlobalConfig
from savanna.tokenizer import CharLevelTokenizer
from savanna.utils import (
    FP8_SHAPE,
    get_ltor_masks_and_position_ids,
    make_upper_case,
    pad_to_multiple,
)


def maybe_fix_sequence_length(global_config, seq_length_padded):
    """
    seq_length_padded is the sequence length after padding AND splitting by context rank

    seq_length will be padded to fp8 requirement 
    AND
    if context parallel, it will be split by context rank

    To run savanna on the model, it needs the padded full sequence length, so we need to adjust the sequence length from the padded, context-sharded sequence length to the full padded sequence length
    """
    seq_length = seq_length_padded * mpu.get_sequence_parallel_world_size()
    global_config.seq_length = seq_length
    return global_config

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
    input_ids, _ = make_upper_case(input_ids)  # This was used during pretraining / extension
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
        attention_mask = (
            mpu.zigzag_split_across_cp_ranks(attention_mask, -2)
            if global_config.materialize_attn_mask
            else None
        )
        position_ids = mpu.zigzag_split_across_cp_ranks(position_ids)

        # Check zig zag reconstruction
        reconstructed_input_ids = mpu.zigzag_gather_from_cp_ranks(input_ids)
        assert torch.allclose(
            original_input_ids, reconstructed_input_ids
        ), f"Zigzag reconstruction failed, {original_input_ids.shape=} != {reconstructed_input_ids.shape=}"

    padded_sharded_seq_length = input_ids.shape[1]
    print_rank_0(f"DEBUG::INFERENCE::prepare_batch {padded_sharded_seq_length=} {global_config.context_parallel_size=} {global_config.seq_length=}")
    global_config = maybe_fix_sequence_length(global_config, padded_sharded_seq_length)
    print_rank_0(f"DEBUG::INFERENCE::prepare_batch::fixed_seq_length {padded_sharded_seq_length=} {global_config.context_parallel_size=} {global_config.seq_length=}")
    
    return (
        input_ids,
        attention_mask if global_config.materialize_attn_mask else None,
        position_ids,
    ), global_config


def pad_to_multiple(d: int, multiple: int = 8) -> int:
    remainder = d % multiple
    if remainder == 0:
        return d, 0
    padding = (multiple - remainder)
    return d + padding, padding

def prepare_inputs(seq, global_config, tokenizer, device="cuda", multiple_to_pad_to: int = 8, padding_value: int = 0):
    input_ids = (
        torch.tensor(
            tokenizer.tokenize(seq),
            dtype=torch.int,
        )
        .to(device)
        .unsqueeze(0)
    )

    # Pad before splitting across cp ranks
    # Each mp rank will see seq length of L // cp_size // mp_size (divide by mp size due to sequence parallelism, cp due to context parallelism)
    # Protocol:
    # 1. Pad the full sequence length to a multiple of 8 according to the seq length that each mp rank will actually see
    #    - This is done by padding the sequence length per mp partition to a multiple of 8, and then multiplying by the mp_size to get total padding
    #    - Pad the full sequence length
    # 2. Split across cp ranks
    # 3. After splitting across cp ranks, the input_ids are zigzagged across mp ranks
    # 4. After calculating logits, we need to gather first BEFORE removing padding, otherwise padding will not be located at end of sequence

    L = input_ids.shape[1]
    num_partitions = mpu.get_model_parallel_world_size() * mpu.get_sequence_parallel_world_size()
    L_per_partition = L // num_partitions

    print_rank_0(f"prepare_inputs before padding: {input_ids.shape=}, {L=} // {num_partitions=} = {L_per_partition}", debug=True)
    L_per_partition_padded, padding_size_per_partition = pad_to_multiple(L_per_partition, multiple_to_pad_to)
    L_padded = L_per_partition_padded * num_partitions
    padding_size = L_padded - L
   
   # print_rank_0(f"prepare_inputs {L=} {L_per_partition=} {L_padded=} {padding_size=} {padding_size_per_partition=}", debug=True)
    
    # padding is inner -> outer, first 2 are seq dim, next 2 are along batch dim
    input_ids = torch.nn.functional.pad(input_ids, (0, padding_size, 0, 0), mode="constant", value=padding_value)
    print_rank_0(f"prepare_inputs after padding {input_ids.shape=} {input_ids[:, -padding_size:]=}", debug=True)

    # if mpu.get_sequence_parallel_world_size() > 1:
    #     original_input_ids = input_ids

    #     input_ids = mpu.zigzag_split_across_cp_ranks(input_ids)
    #     # Check zig zag reconstruction
    #     reconstructed_input_ids = mpu.zigzag_gather_from_cp_ranks(input_ids)
    #     print_rank_0(f"prepare_inputs {reconstructed_input_ids[:, -padding_size:]=}", debug=True)
    #     assert torch.allclose(
    #         original_input_ids, reconstructed_input_ids
    #     ), f"Zigzag reconstruction failed, {original_input_ids.shape=} != {reconstructed_input_ids.shape=}"

    #     print_rank_0(f"prepare_inputs after zigzag split {input_ids.shape=}", debug=True)
    
    return input_ids, padding_size

def generate_seqs(batch_size=1, seq_length=8192, vocab="GATC"):
    from numpy.random import randint

    # Generate random batch size sequences of length seq_length
    seqs = []
    for _ in range(batch_size):
        seqs.append("".join([vocab[i] for i in randint(0, len(vocab), seq_length)]))

    return seqs


