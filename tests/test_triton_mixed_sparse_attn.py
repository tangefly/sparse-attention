import pytest
import torch
from sparse_attn.triton_mixed_sparse_attn import triton_mixed_sparse_attn_fwd

DTYPES = [torch.float16, torch.bfloat16]
HEAD_SIZES = [64, 128]
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64


def full_attn_ref(q, k, v, sm_scale, causal=True):
    q = q.float()
    k = k.float()
    v = v.float()
    B, N, H, D = q.shape
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
    if causal:
        scores = scores + torch.triu(
            torch.full((N, N), float("-inf"), device=q.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)


def make_full_no_vert(B, H, N, device="cuda"):
    NR = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    NS = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    block_count = torch.full((B, H, NR), NS, dtype=torch.int32, device=device)
    column_count = torch.zeros(B, H, NR, dtype=torch.int32, device=device)
    block_offset = (
        torch.arange(NS, device=device, dtype=torch.int32) * BLOCK_SIZE_N
    ).reshape(1, 1, 1, NS).expand(B, H, NR, NS).contiguous()
    column_index = torch.zeros(B, H, NR, 1, dtype=torch.int32, device=device)
    return block_count, block_offset, column_count, column_index


def make_sparse_indices(B, H, N, NNZ_S, device="cuda"):
    NR = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    NS = min(NNZ_S, N // BLOCK_SIZE_N)
    NV = N - NS * BLOCK_SIZE_N
    if NS == 0:
        NV = N
    block_count = torch.full((B, H, NR), NS, dtype=torch.int32, device=device)
    column_count = torch.full((B, H, NR), NV, dtype=torch.int32, device=device)
    block_offset = (
        torch.arange(max(NS, 1), device=device, dtype=torch.int32)[:NS] * BLOCK_SIZE_N
    ).reshape(1, 1, 1, NS).expand(B, H, NR, NS).contiguous()
    column_index = (
        torch.arange(max(NV, 1), device=device, dtype=torch.int32)[:NV] + NS * BLOCK_SIZE_N
    ).reshape(1, 1, 1, NV).expand(B, H, NR, NV).contiguous()
    return block_count, block_offset, column_count, column_index


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("seqlen", [64, 128, 256])
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_full_coverage_no_vertical(
    batch_size, num_heads, seqlen, head_size, dtype, causal
):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    sm_scale = head_size ** -0.5
    B, H, N, D = batch_size, num_heads, seqlen, head_size
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    bc, bo, cc, ci = make_full_no_vert(B, H, N)
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    ref = full_attn_ref(q, k, v, sm_scale, causal=causal).to(dtype=dtype)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal,
    )
    out_bqhd = out.permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(out_bqhd, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("NNZ_S", [0, 1, 2, 3])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_sparse_pattern_smoke(
    batch_size, num_heads, seqlen, NNZ_S, head_size, dtype, causal
):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    sm_scale = head_size ** -0.5
    B, H, N, D = batch_size, num_heads, seqlen, head_size
    if NNZ_S * BLOCK_SIZE_N > N:
        pytest.skip("NNZ_S too large for seqlen")
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    bc, bo, cc, ci = make_sparse_indices(B, H, N, NNZ_S)
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal,
    ).permute(0, 2, 1, 3)
    assert out.shape == (B, N, H, D)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen", [100, 200])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_partial_last_block(batch_size, seqlen, head_size, dtype):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    sm_scale = head_size ** -0.5
    B, H, N, D = batch_size, 2, seqlen, head_size
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    ref = full_attn_ref(q, k, v, sm_scale, causal=False).to(dtype=dtype)
    bc, bo, cc, ci = make_full_no_vert(B, H, N)
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=False,
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("seqlen", [256])
@torch.inference_mode()
def test_variable_seqlen(seqlen):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    dtype = torch.float16
    sm_scale = 128 ** -0.5
    B, H, N, D = 2, 2, seqlen, 128
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    bc, bo, cc, ci = make_full_no_vert(B, H, N)
    seqlens = torch.tensor([150, 256], dtype=torch.int32, device="cuda")
    ref = full_attn_ref(q, k, v, sm_scale, causal=True).to(dtype=dtype)
    ref[0, 150:] = 0.0
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=True,
    ).permute(0, 2, 1, 3)
    assert not torch.isnan(out).any()
    assert out[0, 150:].abs().max().item() < 1e-6
    torch.testing.assert_close(out[0, :150], ref[0, :150], atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_no_slash(batch_size, seqlen, dtype):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    sm_scale = 128 ** -0.5
    B, H, N, D = batch_size, 2, seqlen, 128
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    NR = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    bc = torch.zeros(B, H, NR, dtype=torch.int32)
    bo = torch.zeros(B, H, NR, 1, dtype=torch.int32)
    cc = torch.full((B, H, NR), N, dtype=torch.int32)
    ci = torch.arange(N, device="cuda", dtype=torch.int32).reshape(
        1, 1, 1, N
    ).expand(B, H, NR, N).contiguous()
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    ref = full_attn_ref(q, k, v, sm_scale, causal=False).to(dtype=dtype)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=False,
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_no_vertical(batch_size, seqlen, dtype):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    sm_scale = 128 ** -0.5
    B, H, N, D = batch_size, 2, seqlen, 128
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    bc, bo, cc, ci = make_full_no_vert(B, H, N)
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    ref = full_attn_ref(q, k, v, sm_scale, causal=True).to(dtype=dtype)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=True,
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_vs_naive(batch_size, seqlen, head_size, dtype, causal):
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    from sparse_attn.naive_mixed_sparse_attn import naive_mixed_sparse_attn_fwd
    sm_scale = head_size ** -0.5
    B, H, N, D = batch_size, 2, seqlen, head_size
    q = torch.randn(B, N, H, D, dtype=dtype)
    k = torch.randn(B, N, H, D, dtype=dtype)
    v = torch.randn(B, N, H, D, dtype=dtype)
    bc, bo, cc, ci = make_full_no_vert(B, H, N)
    seqlens = torch.tensor([N] * B, dtype=torch.int32)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out_triton = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal,
    )
    out_naive = naive_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal,
    )
    torch.testing.assert_close(out_triton, out_naive, atol=2e-2, rtol=1e-2)


@torch.inference_mode()
def test_vs_cuda():
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    from sparse_attn import sparse_attn_func
    B, H, N, D = 1, 1, 128, 128
    sm_scale = D ** -0.5
    q = torch.randn(B, N, H, D, dtype=torch.float16)
    k = torch.randn(B, N, H, D, dtype=torch.float16)
    v = torch.randn(B, N, H, D, dtype=torch.float16)
    bc, bo, cc, ci = make_sparse_indices(B, H, N, 2)
    seqlens = torch.tensor([N], dtype=torch.int32)
    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()
    out_triton = triton_mixed_sparse_attn_fwd(
        q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
        sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=False,
    ).permute(0, 2, 1, 3)
    out_cuda, _ = sparse_attn_func(
        q, k, v, bc, bo, cc, ci, causal=False, return_softmax_lse=True,
    )
    torch.testing.assert_close(out_triton, out_cuda, atol=2e-2, rtol=1e-2)
