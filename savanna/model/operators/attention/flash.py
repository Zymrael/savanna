# Based on: https://github.com/HazyResearch/flash-attention/blob/4a6eaa9f27df6fff7ffb2c24e894938a687dd870/flash_attn/flash_attn_interface.py
import flash_attn
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import (
    flash_attn_qkvpacked_func,
    flash_attn_triton,
    flash_attn_varlen_qkvpacked_func,
)

from savanna import mpu, print_rank_0


def flash_attn_unpadded_unpacked_func_triton(q, k, v, bias=None, causal=False, softmax_scale=None):
    return flash_attn_triton.flash_attn_func(q, k, v, bias, causal, softmax_scale)


def _flash_attn_forward_cuda(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    return_softmax,
    num_splits=0,
    generator=None,
):
    """
    num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
    it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
    Don't change it unless you know what you're doing.
    """
    softmax_lse, *rest = flash_attn.fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        return_softmax,
        num_splits,
        generator,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    S_dmask = rest[0] if return_softmax else None
    return out, softmax_lse, S_dmask


def _flash_attn_backward_cuda(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    num_splits=0,
    generator=None,
):
    """
    num_splits: whether to parallelize over the seqlen_k dimension (num_splits > 1) or
    not (num_splits = 1). num_splits=0 means it will be set by an internal heuristic.
    Any value above 1 will call the same kernel (i.e. num_splits=2 would call the same kernel
    as num_splits=3), so effectively the choices are 0, 1, and 2.
    This hyperparameter can be tuned for performance, but default value (heuristic) should work fine.
    """
    _, _, _, softmax_d = flash_attn.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        num_splits,
        generator,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask = _flash_attn_forward_cuda(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            torch.empty_like(qkv[:, 0]),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        ctx.save_for_backward(qkv, out, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward_cuda(
            dout,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None


def flash_attn_unpadded_qkvpacked_func_cuda(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    return FlashAttnQKVPackedFunc.apply(
        qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal, return_attn_probs
    )


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask = _flash_attn_forward_cuda(
            q,
            kv[:, 0],
            kv[:, 1],
            torch.empty_like(q),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        ctx.save_for_backward(q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        _flash_attn_backward_cuda(
            dout,
            q,
            kv[:, 0],
            kv[:, 1],
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dq, dkv, None, None, None, None, None, None, None, None


def flash_attn_unpadded_kvpacked_func_cuda(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_attn_probs,
    )


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask = _flash_attn_forward_cuda(
            q,
            k,
            v,
            torch.empty_like(q),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward_cuda(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dq, dk, dv, None, None, None, None, None, None, None, None


def flash_attn_unpadded_func_cuda(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_attn_probs,
    )


class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )


class RingAttention(nn.Module):
    def __init__(self, global_config, causal, num_attention_heads_per_partition, hidden_size_per_attention_head, pos_emb, alibi_embed, dropout_p):
        super().__init__()
        from ring_flash_attn.zigzag_ring_flash_attn_varlen import (
            zigzag_ring_flash_attn_varlen_qkvpacked_func,
        )
        print_rank_0(f"DEBUG::ParallelSequenceMixer::use_ring_attention: {mpu.get_sequence_parallel_world_size()=} {global_config.use_cp_ring=}")
        self.attn = zigzag_ring_flash_attn_varlen_qkvpacked_func
        self.causal = causal
        self.num_attention_heads_per_partition = num_attention_heads_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.pos_emb = pos_emb
        self.alibi_embed = alibi_embed
        self.dropout_p = dropout_p
        self.global_config = global_config
       
    def forward(self, query_layer, key_layer, value_layer):
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        batch_size = output_size[0]
        max_seqlen_q = output_size[2]
        max_seqlen_k = output_size[3]

        # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
        key_layer = key_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )
        value_layer = value_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )

        # [sq, b, np, hn] -> [b * sq, 1, np, hn]
        query_layer = query_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[2], 1, output_size[1], -1
        )

        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_q,
            step=max_seqlen_q,
            dtype=torch.int32,
            device=query_layer.device,
        )

        # Combined q/k/v into [b * s, 3, np, hn].
        qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

        # only pass in alibi_slopes kwarg] if we use AliBi.
        # Flash attn defaults to (-1,-1), or
        # does not have this kwarg prior to v2.3.0
        extra_kwargs = {}
        if self.pos_emb == "alibi":
            extra_kwargs["alibi_slopes"] = self.alibi_embed.slopes.to(
                query_layer.device
            ).to(torch.float32)

        output = self.attn(
            qkv,
            cu_seqlens_q,
            max_seqlen_q,
            self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=self.causal,
            group=mpu.get_sequence_parallel_group(),
            **extra_kwargs,
        )

        output = rearrange(output, "(b l) h dh -> b l (h dh)",
                           b=batch_size, l=max_seqlen_q, 
                           h=self.num_attention_heads_per_partition, dh=self.hidden_size_per_attention_head)
        return output       