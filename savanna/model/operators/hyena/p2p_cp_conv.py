import torch
from torch.autograd import Function
import torch.distributed as dist


def zigzag_get_overlapping_patches(data, seq_dim, overlap_size):
    """
    Extracts the overlapping patches from data in each rank.
    Arguments:
        data (torch.Tensor): The concatenated data (chunk_a and chunk_b), e.g., [0, 3] , [1, 2] with zigzag and 2 GPUs.
        seq_dim (int): The sequence dimension along which the data is concatenated.
        overlap_size (int): The size of the overlapping patch.
    Returns:
        overlap_a, overlap_b (torch.Tensor): The overlapping chunks from the data. That is the end of the lowest, and
        the beginning of the last, e.g., end for 0 and start for 3.
    """
    assert seq_dim >= 0, "Negative indexes not supported."

    data_shape = list(data.shape)
    modified_shape = list(data.shape)
    modified_shape[seq_dim: seq_dim + 1] = [2, data_shape[seq_dim] // 2]

    reshaped_data = torch.reshape(data, modified_shape)

    # Move the dimension of the chunks to the first position
    # Create a permutation where seq_dim is moved to position 0
    permute_order = list(range(len(reshaped_data.shape)))
    permute_order.insert(0, permute_order.pop(seq_dim))  # Move seq_dim to index 0

    reshaped_data = reshaped_data.permute(dims=permute_order)

    seq_len = reshaped_data.shape[seq_dim + 1]  # Remember that a new dimension was added.
    overlapping_patches = reshaped_data.narrow(dim=seq_dim + 1, start=seq_len-overlap_size, length=overlap_size) # Last n elements.
    return overlapping_patches[0], overlapping_patches[1]


class ExchangeOverlappingRegionsCausal(Function):
    """
        A custom autograd function for exchanging overlapping regions between chunks of data in a causal manner.
        The data is split across multiple GPUs using a distributed process group.
        The forward method handles the exchange of overlapping regions between chunks, while the backward method computes the gradients.
        Attributes:
        - ctx: A context object that stores information for the forward and backward passes.
        - chunk_a: Chunk to pass to the left.
        - chunk_b: Chunk to pass to the right.
        - group: The CP group
        - group_rank: The rank in the cp_group.
        """

    @staticmethod
    def forward(ctx, chunk_a, chunk_b, group, group_rank):

        group_ranks = dist.get_process_group_ranks(group)  # Get all global ranks in the cp_group
        group_world_size = len(group_ranks)  # Size of the cp_group

        ctx.group = group
        ctx.group_rank = group_rank
        ctx.group_world_size = group_world_size
        ctx.group_ranks = group_ranks

        # Initialize requests
        reqs = []

        # Exchange overlaps for chunk_a
        if group_rank > 0:
            # Receive overlap from previous rank
            recv_shape = list(chunk_a.shape)
            recv_prev_a = torch.empty(recv_shape, dtype=chunk_a.dtype, device=chunk_a.device)
            req_recv_a = dist.irecv(recv_prev_a, src=group_ranks[group_rank - 1])
            reqs.append(req_recv_a)
        else:
            recv_prev_a = None

        if group_rank < group_world_size - 1:
            # Send overlap to next rank
            req_send_a = dist.isend(chunk_a.contiguous(), dst=group_ranks[group_rank + 1])
            reqs.append(req_send_a)

        # Exchange overlaps for chunk_b
        if group_rank < group_world_size - 1:
            # Receive overlap from next rank
            recv_shape = list(chunk_b.shape)
            recv_next_b = torch.empty(recv_shape, dtype=chunk_b.dtype, device=chunk_b.device)
            req_recv_b = dist.irecv(recv_next_b, src=group_ranks[group_rank + 1])
            reqs.append(req_recv_b)
        else:
            recv_next_b = None

        if group_rank > 0:
            # Send overlap to previous rank
            req_send_b = dist.isend(chunk_b.contiguous(), dst=group_ranks[group_rank - 1])
            reqs.append(req_send_b)

        # Wait for all communication to finish
        for req in reqs:
            req.wait()

        # If no chunks received, use zeros instead (for consistency)
        if recv_prev_a is None:
            recv_prev_a = torch.zeros_like(chunk_a, dtype=chunk_a.dtype, device=chunk_a.device)
        if recv_next_b is None:
            recv_next_b = chunk_a.clone().contiguous()  # Got to receive from the same rank, but previous split.

        return recv_prev_a, recv_next_b

    @staticmethod
    def backward(ctx, grad_chunk_a, grad_chunk_b):
        # chunk_a, chunk_b = ctx.saved_tensors
        group = ctx.group
        group_rank = ctx.group_rank
        group_world_size = ctx.group_world_size
        group_ranks = ctx.group_ranks

        # Initialize gradients with zeros
        _grad_chunk_a = torch.zeros_like(grad_chunk_a)
        _grad_chunk_b = torch.zeros_like(grad_chunk_b)

        # Initialize requests
        reqs = []

        ### Handling grad_chunk_a

        # If rank > 0, send grad_recv_prev_a to rank - 1
        if group_rank > 0:
            req_send_a = dist.isend(grad_chunk_a.contiguous(), dst=group_ranks[group_rank - 1])
            reqs.append(req_send_a)
        else:
            # At rank 0, there's no previous rank to receive from, so we only consider local gradient contributions
            pass  # No action needed

        # If rank < world_size - 1, receive grad_chunk_a from rank + 1
        if group_rank < group_world_size - 1:
            grad_chunk_a_recv = torch.empty_like(grad_chunk_a)
            req_recv_a = dist.irecv(grad_chunk_a_recv, src=group_ranks[group_rank + 1])
            reqs.append(req_recv_a)

        ### Handling grad_chunk_b

        # If rank < world_size - 1, send grad_recv_next_b to rank + 1
        if group_rank < group_world_size - 1:
            req_send_b = dist.isend(grad_chunk_b.contiguous(), dst=group_ranks[group_rank + 1])
            reqs.append(req_send_b)

        # If rank > 0, receive grad_chunk_b from rank - 1
        if group_rank > 0:
            grad_chunk_b_recv = torch.empty_like(grad_chunk_b)
            req_recv_b = dist.irecv(grad_chunk_b_recv, src=group_ranks[group_rank - 1])
            reqs.append(req_recv_b)

        # Wait for all communication to finish
        for req in reqs:
            req.wait()

        # Add received gradients
        if group_rank < group_world_size - 1:
            _grad_chunk_a = grad_chunk_a_recv

        if group_rank > 0:
            _grad_chunk_b = grad_chunk_b_recv

        if group_rank == group_world_size - 1:
            _grad_chunk_a = grad_chunk_b  # In the last split, the chunks are exchanged locally.

        return _grad_chunk_a, _grad_chunk_b, None, None, None


if __name__ == "__main__":
    # Define a small test dataset
    data = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],  # First chunk (chunk_a)
        [9, 10, 11, 12, 13, 14, 15, 16]  # Second chunk (chunk_b)
    ]).flatten()
    print(f"Data:\n{data}")

    overlap_a, overlap_b = zigzag_get_overlapping_patches(data, seq_dim=0, overlap_size=3)

    # Print the result to verify
    print("Overlap from the end of the first chunk (overlap_a):")
    print(overlap_a)

    print("\nOverlap from the beginning of the second chunk (overlap_b):")
    print(overlap_b)