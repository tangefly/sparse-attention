import torch
from torch import Tensor
from sparse_attn_cuda import add, mha_fwd_sparse, _convert_vertical_slash_indexes

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def cadd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    CUDA 加法，确保输入是 CUDA Tensor 并且形状匹配。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected x to be torch.Tensor, got {type(x)}")
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"Expected y to be torch.Tensor, got {type(y)}")

    if x.device.type != "cuda":
        raise ValueError(f"x must be on CUDA device, got {x.device}")
    if y.device.type != "cuda":
        raise ValueError(f"y must be on CUDA device, got {y.device}")

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    if x.dtype != y.dtype:
        raise ValueError(f"x and y must have the same dtype, got {x.dtype} vs {y.dtype}")
    
    if x.dtype != torch.float32:
        raise ValueError(f"x must be float32, got {x.dtype}")
    
    if y.dtype != torch.float32:
        raise ValueError(f"y must be float32, got {y.dtype}")

    return add(x, y)

def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = mha_fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def convert_vertical_slash_indexes(
    seqlens: Tensor,
    vertical_indexes: Tensor,
    slash_indexes: Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int
):
    block_count, block_offset, column_count, column_index = _convert_vertical_slash_indexes(
        seqlens, 
        vertical_indexes, 
        slash_indexes, 
        context_size,
        block_size_M,
        block_size_N
    )

    return block_count, block_offset, column_count, column_index
