import pytest
import torch
from typing import Tuple
from sparse_attn import sparse_attn_func

DTYPES = [torch.bfloat16]

def ref_attn(q, k, v, softmax_scale=None, is_causal=False):
    """
    Dense reference attention (用于对比 sparse kernel)

    输入:
        q: (B, Sq, H, D)
        k: (B, Sk, H, D)
        v: (B, Sk, H, D)

    输出:
        out: (B, Sq, H, D)
        softmax_lse: (B, H, Sq)
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)
    
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 1, 3).contiguous()
    v_ = v.permute(0, 2, 1, 3).contiguous()

    scores = torch.matmul(q_, k_.transpose(-2, -1)) * softmax_scale

    if is_causal:
        mask = torch.triu(
            torch.ones(Sq, Sk, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    softmax_lse = torch.logsumexp(scores, dim=-1)  # (B,H,Sq)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_)
    out = out.permute(0, 2, 1, 3).contiguous()  # (B,Sq,H,D)

    return out, softmax_lse

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_lens", [(1, 2048), (1023, 2049)])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("NNZ_S", [2, 3])
@torch.inference_mode()
def test_sparse_attention(
        batch_size: int,
        seq_lens: Tuple[int, int],
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        NNZ_S: int,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    block_size_M = 64
    block_size_N = 64
    seqlen_q, seqlen_k = seq_lens
    q = torch.randn(
        batch_size, seqlen_q, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    NUM_ROWS = (seqlen_q + block_size_M - 1) // block_size_M
    if NNZ_S * block_size_N > seqlen_k:
        return
    NNZ_V = seqlen_k - NNZ_S * block_size_N
    block_count = torch.tensor([NNZ_S] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS)
    column_count = torch.tensor([NNZ_V] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS)
    block_offset = torch.tensor([[i * block_size_N for i in range(NNZ_S)]] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
    column_index = torch.tensor([[NNZ_S * block_size_N + i for i in range(NNZ_V)]] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)
    from sparse_attn import sparse_attn_func
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = ref_attn(q, k, v)

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(out - ref_out))}"
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(lse - ref_lse))}"