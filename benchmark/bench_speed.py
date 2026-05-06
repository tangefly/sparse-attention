import time
import torch
from sparse_attn import sparse_attn_func
from sparse_attn.naive_mixed_sparse_attn import naive_mixed_sparse_attn_fwd
from sparse_attn.triton_mixed_sparse_attn import triton_mixed_sparse_attn_fwd

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64


def make_sparse_indices(B, H, N, NNZ_S, device="cuda"):
    NR = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    NS = min(NNZ_S, N // BLOCK_SIZE_N)
    NV = N - NS * BLOCK_SIZE_N
    if NS == 0:
        NV = N
    bc = torch.full((B, H, NR), NS, dtype=torch.int32, device=device)
    cc = torch.full((B, H, NR), NV, dtype=torch.int32, device=device)
    bo = (torch.arange(max(NS, 1), device=device, dtype=torch.int32)[:NS] * BLOCK_SIZE_N
          ).reshape(1, 1, 1, NS).expand(B, H, NR, NS).contiguous()
    ci = (torch.arange(max(NV, 1), device=device, dtype=torch.int32)[:NV] + NS * BLOCK_SIZE_N
          ).reshape(1, 1, 1, NV).expand(B, H, NR, NV).contiguous()
    return bc, bo, cc, ci


@torch.inference_mode()
def bench_one(B, H, N, D, NNZ_S, causal, warmup, iters):
    device = "cuda"
    sm_scale = D ** -0.5

    q = torch.randn(B, N, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device=device)
    v = torch.randn(B, N, H, D, dtype=torch.float16, device=device)
    seqlens = torch.tensor([N] * B, dtype=torch.int32, device=device)
    bc, bo, cc, ci = make_sparse_indices(B, H, N, NNZ_S, device)

    q_bkhd = q.permute(0, 2, 1, 3).contiguous()
    k_bkhd = k.permute(0, 2, 1, 3).contiguous()
    v_bkhd = v.permute(0, 2, 1, 3).contiguous()

    for _ in range(warmup):
        naive_mixed_sparse_attn_fwd(q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
                                    sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal)
        triton_mixed_sparse_attn_fwd(q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
                                     sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal)
        sparse_attn_func(q, k, v, bc, bo, cc, ci, causal=causal)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        naive_mixed_sparse_attn_fwd(q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
                                    sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal)
    torch.cuda.synchronize()
    t_naive = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        triton_mixed_sparse_attn_fwd(q_bkhd, k_bkhd, v_bkhd, seqlens, bc, bo, cc, ci,
                                     sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        sparse_attn_func(q, k, v, bc, bo, cc, ci, causal=causal)
    torch.cuda.synchronize()
    t_cuda = (time.perf_counter() - start) / iters

    return t_cuda, t_triton, t_naive


def main():
    configs = [
        (1, 4, 1024, 64, 8),
        (1, 4, 1024, 128, 8),
        (1, 8, 2048, 128, 16),
        (4, 4, 1024, 128, 8),
    ]
    warmup, iters = 5, 20
    print(f"{'B':>3} {'H':>3} {'N':>6} {'D':>4} {'NS':>5}  {'CUDA(ms)':>9} {'Triton(ms)':>10} {'Naive(ms)':>10} {'T/C':>7} {'T/N':>7}")
    print("-" * 80)
    for B, H, N, D, NNZ_S in configs:
        tc, tt, tn = bench_one(B, H, N, D, NNZ_S, causal=True, warmup=warmup, iters=iters)
        print(f"{B:>3} {H:>3} {N:>6} {D:>4} {NNZ_S:>5}  {tc*1000:>9.3f} {tt*1000:>10.3f} {tn*1000:>10.3f} {tn/tc:>6.1f}x {tn/tt:>6.1f}x")

    tc, tt, tn = bench_one(1, 8, 2048, 128, 16, causal=False, warmup=warmup, iters=iters)
    print(f"\nnon-causal: CUDA={tc*1000:.3f}ms  Triton={tt*1000:.3f}ms  Naive={tn*1000:.3f}ms")


if __name__ == "__main__":
    main()
