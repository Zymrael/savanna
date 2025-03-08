import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple, List, Literal
from einops import rearrange
from torch.autograd.function import Function


def _get_zigzag_indices(N, device=None):
    """
    Generates the zigzag indices for rearrangement.
    Args:
        N (int): The total number of chunks.
        device (torch.device): The device on which to create tensors.
    Returns:
        torch.Tensor: The zigzag indices.
    """
    half_N = (N + 1) // 2
    idx1 = torch.arange(half_N, device=device)
    idx2 = torch.arange(N - 1, half_N - 1, -1, device=device)
    zigzag_idx = torch.empty(N, dtype=torch.long, device=device)
    zigzag_idx[0::2] = idx1
    zigzag_idx[1::2] = idx2
    return zigzag_idx


def _get_inverse_zigzag_indices(N, device=None):
    """
    Generates the inverse zigzag indices for rearrangement.
    Args:
        N (int): The total number of chunks.
        device (torch.device): The device on which to create tensors.
    Returns:
        torch.Tensor: The inverse zigzag indices.
    """
    half_N = N // 2
    idx1 = torch.arange(half_N, device=device)
    idx2 = torch.arange(N - 1, half_N - 1, -1, device=device)
    zigzag_idx = torch.empty(N, dtype=torch.long, device=device)
    zigzag_idx[0::2] = idx1
    zigzag_idx[1::2] = idx2
    inverse_zigzag_idx = torch.argsort(zigzag_idx)
    return inverse_zigzag_idx


def all_to_all_single_fn(
    group: dist.ProcessGroup,
    type: Literal["split_to_full", "full_to_split"],
    input: torch.Tensor,
    with_zigzag_splitting: bool = True
) -> torch.Tensor:
    """
    Autograd-aware all_to_all_single communication function.
    Args:
        group (dist.ProcessGroup): The process group for communication.
        type (str): Either 'split_to_full' or 'full_to_split' to specify the communication pattern.
        input (torch.Tensor): Input tensor to be communicated.
        with_zigzag_splitting (bool, optional): Whether to apply zigzag splitting. Defaults to True.
    Returns:
        torch.Tensor: Output tensor after communication.
    """

    world_size = dist.get_world_size(group=group)

    if type == "split_to_full":
        """Given an split sequence, it gathers the whole sequence, while splitting across the channels dimension."""

        B, D, l = input.shape
        L = l * world_size
        d = D // world_size

        # Reshape and permute input for communication
        input_reshaped = rearrange(input, "B (cp d) l -> cp B d l", cp=world_size).contiguous() # [cp_world_size, B, d, l]

        # Perform all_to_all_single communication
        output_reshaped = torch.empty_like(input_reshaped)
        dist.all_to_all_single(output_reshaped, input_reshaped, group=group) # [cp_world_size, B, d, l]

        # Permute and reshape output back to original form
        output = rearrange(output_reshaped, "cp B d l -> B d (cp l)", cp=world_size).contiguous()

        if with_zigzag_splitting:
            num_chunks = 2 * world_size
            unzigzagged_split_length = L // num_chunks  # Length of each small chunk
            device = output.device
            inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks, device=device)

            # Vectorized rearrangement using inverse zigzag indices
            output = output.reshape(B, d, num_chunks, unzigzagged_split_length).index_select(
                dim=-2, index=inverse_zigzag_idx).reshape(B, d, L)

        return output

    elif type == "full_to_split":
        """Given a full sequence split across channels, splits across the sequence length and while gathering the channels."""

        B, d, L = input.shape
        l = L // world_size
        D = d * world_size

        if with_zigzag_splitting:
            num_chunks = 2 * world_size
            chunk_length = L // num_chunks  # Length of each small chunk
            device = input.device
            zigzag_idx = _get_zigzag_indices(num_chunks, device=device)

            # Ensure L is divisible by num_chunks
            if L % num_chunks != 0:
                raise ValueError(f"Sequence length {L} is not divisible by num_chunks {num_chunks}")

            # Vectorized rearrangement using zigzag indices
            input = input.reshape(B, d, num_chunks, chunk_length).index_select(dim=-2, index=zigzag_idx).reshape(B, d, L)

        # Reshape and permute inputs for communication
        input_reshaped = rearrange(input, "b d (cp l) -> cp b d l", cp=world_size).contiguous() # [cp_world_size, b, d, l]

        # Perform all_to_all_single communication
        output_reshaped = torch.empty_like(input_reshaped)
        dist.all_to_all_single(output_reshaped, input_reshaped, group=group) # [cp_world_size, B, d, l]

        # Permute and reshape outputs back to original form
        output = rearrange(output_reshaped, "cp b d l -> b (cp d) l", cp=world_size).contiguous()

        return output

    else:
        raise ValueError(f"Unknown type {type}")


class AllToAllSingleFunction(Function):
    """
        A custom autograd function for performing all_to_all_single communication with optional zigzag splitting.
        Attributes:
        - ctx: A context object that stores information for the forward and backward passes.
        - group: The process group for communication.
        - type: The type of communication pattern ('split_to_full' or 'full_to_split').
        - with_zigzag_splitting: A boolean indicating whether to apply zigzag splitting.
        """

    @staticmethod
    def forward(ctx, input_tensor, group, type, with_zigzag_splitting):
        ctx.group = group
        ctx.type = type
        ctx.with_zigzag_splitting = with_zigzag_splitting

        # Detach input_tensor to prevent PyTorch from tracking operations inside the communication
        input_tensor = input_tensor.detach()

        # Perform the communication operation
        output = all_to_all_single_fn(
            group=ctx.group,
            type=ctx.type,
            input=input_tensor,
            with_zigzag_splitting=ctx.with_zigzag_splitting
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # The backward pass will perform the reverse communication
        grad_input = all_to_all_single_fn(
            group=ctx.group,
            type="split_to_full" if ctx.type != "split_to_full" else "full_to_split",
            input=grad_output,
            with_zigzag_splitting=ctx.with_zigzag_splitting
        )
        # Return the gradient w.r.t. the input_tensor and None for other arguments
        return grad_input, None, None, None


if __name__ == "__main__":
    """
    Test function to verify that the zigzag splitting and inverse zigzag rearrangement work correctly.
    """
    # Parameters
    B = 1  # Batch size
    d = 1  # Number of features/channels
    L = 8  # Sequence length
    cp_world_size = 4  # Context parallel world size
    num_chunks = 2 * cp_world_size  # Total number of chunks for zigzag splitting

    # Create an input tensor with sequential values
    input_tensor = torch.arange(L).reshape(B, d, L).float()

    seq_chunks = torch.chunk(input_tensor, 2 * cp_world_size, dim=-1)
    _data = [
        torch.cat((seq_chunks[i], seq_chunks[-(i + 1)]), dim=-1)
        for i in range(cp_world_size)
    ]
    print(_data)

    print("Original tensor:")
    print(input_tensor)

    # Generate zigzag indices
    zigzag_idx = _get_zigzag_indices(num_chunks)
    print("\nZigzag indices:")
    print(zigzag_idx)

    # Apply zigzag splitting
    chunk_length = L // num_chunks
    input_reshaped = input_tensor.reshape(B, d, num_chunks, chunk_length)
    zigzag_tensor = input_reshaped.index_select(dim=2, index=zigzag_idx).reshape(B, d, L)

    print("\nTensor after zigzag splitting:")
    print(zigzag_tensor)

    # Generate inverse zigzag indices
    inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks)
    print("\nInverse zigzag indices:")
    print(inverse_zigzag_idx)

    # Apply inverse zigzag rearrangement
    zigzag_reshaped = zigzag_tensor.reshape(B, d, num_chunks, chunk_length)
    recovered_tensor = zigzag_reshaped.index_select(dim=2, index=inverse_zigzag_idx).reshape(B, d, L)

    print("\nRecovered tensor after inverse zigzag rearrangement:")
    print(recovered_tensor)

    # Verify that the recovered tensor matches the original tensor
    if torch.allclose(input_tensor, recovered_tensor):
        print("\nSuccess: Recovered tensor matches the original tensor.")
    else:
        print("\nFailure: Recovered tensor does not match the original tensor.")