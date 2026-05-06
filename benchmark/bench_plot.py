import time
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from sparse_attn import sparse_attn_func
from sparse_attn.naive_mixed_sparse_attn import naive_mixed_sparse_attn_fwd
from sparse_attn.triton_mixed_sparse_attn import triton_mixed_sparse_attn_fwd

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
DEVICE = "cuda"
DTYPE = torch.float16
OUT_DIR = Path(__file__).parent


def make_sparse_indices(B, H, N, NNZ_S, device=DEVICE):
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


def time_one(fn, *args, warmup=3, iters=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


@torch.inference_mode()
def bench_config(B, H, N, D, NNZ_S, causal=True, warmup=3, iters=10):
    sm_scale = D ** -0.5
    seqlens = torch.tensor([N] * B, dtype=torch.int32, device=DEVICE)
    bc, bo, cc, ci = make_sparse_indices(B, H, N, NNZ_S, DEVICE)

    q_bkhd = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    k_bkhd = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
    v_bkhd = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)

    q_bnsh = q_bkhd.permute(0, 2, 1, 3).contiguous()
    k_bnsh = k_bkhd.permute(0, 2, 1, 3).contiguous()
    v_bnsh = v_bkhd.permute(0, 2, 1, 3).contiguous()

    t_naive = time_one(
        naive_mixed_sparse_attn_fwd, q_bkhd, k_bkhd, v_bkhd, seqlens,
        bc, bo, cc, ci, sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal,
        warmup=warmup, iters=iters,
    )

    t_triton = time_one(
        triton_mixed_sparse_attn_fwd, q_bkhd, k_bkhd, v_bkhd, seqlens,
        bc, bo, cc, ci, sm_scale, BLOCK_SIZE_M, BLOCK_SIZE_N, causal=causal,
        warmup=warmup, iters=iters,
    )

    t_cuda = time_one(
        sparse_attn_func, q_bnsh, k_bnsh, v_bnsh, bc, bo, cc, ci,
        causal=causal, warmup=warmup, iters=iters,
    )

    return {"CUDA": t_cuda, "Triton": t_triton, "Naive": t_naive}


def set_style(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def run_sweep_seqlen():
    """Sweep sequence length N."""
    B, H, D, NNZ_S = 1, 8, 128, 16
    Ns = [512, 1024, 2048, 4096, 8192]
    results = []
    for N in Ns:
        r = bench_config(B, H, N, D, NNZ_S, warmup=2, iters=10)
        results.append(r)
        print(f"  N={N:5d}  CUDA={r['CUDA']*1000:.3f}ms  Triton={r['Triton']*1000:.3f}ms  Naive={r['Naive']*1000:.3f}ms")
    return Ns, results


def run_sweep_sparsity():
    """Sweep slash sparsity NNZ_S."""
    B, H, N, D = 1, 8, 2048, 128
    NNZ_Ss = [4, 8, 16, 32, 64]
    results = []
    for NNZ_S in NNZ_Ss:
        r = bench_config(B, H, N, D, NNZ_S, warmup=2, iters=10)
        results.append(r)
        print(f"  NS={NNZ_S:3d}  CUDA={r['CUDA']*1000:.3f}ms  Triton={r['Triton']*1000:.3f}ms  Naive={r['Naive']*1000:.3f}ms")
    return NNZ_Ss, results


def run_sweep_batch():
    """Sweep batch size B."""
    H, N, D, NNZ_S = 4, 1024, 128, 8
    Bs = [1, 2, 4, 8]
    results = []
    for B in Bs:
        r = bench_config(B, H, N, D, NNZ_S, warmup=2, iters=10)
        results.append(r)
        print(f"  B={B:2d}  CUDA={r['CUDA']*1000:.3f}ms  Triton={r['Triton']*1000:.3f}ms  Naive={r['Naive']*1000:.3f}ms")
    return Bs, results


def run_sweep_headdim():
    """Sweep head dimension D."""
    B, H, N, NNZ_S = 1, 8, 2048, 16
    Ds = [64, 128]
    results = []
    for D in Ds:
        r = bench_config(B, H, N, D, NNZ_S, warmup=2, iters=10)
        results.append(r)
        print(f"  D={D:3d}  CUDA={r['CUDA']*1000:.3f}ms  Triton={r['Triton']*1000:.3f}ms  Naive={r['Naive']*1000:.3f}ms")
    return Ds, results


def plot_results(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mixed Sparse Flash Attention: Performance Comparison", fontsize=16, y=1.02)

    colors = {"CUDA": "#E74C3C", "Triton": "#3498DB", "Naive": "#95A5A6"}
    markers = {"CUDA": "o", "Triton": "s", "Naive": "^"}
    impls = ["CUDA", "Triton", "Naive"]

    def fill(ax, xs, label, data_dicts, xlabel, ylabel, title):
        lines = []
        for impl in impls:
            ys = [d[impl] * 1000 for d in data_dicts]
            (line,) = ax.plot(xs, ys, label=impl, color=colors[impl],
                              marker=markers[impl], linewidth=1.8, markersize=6)
            lines.append(line)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        return lines

    # (a) Latency vs Sequence Length
    ax0 = axes[0, 0]
    fill(ax0, all_data["seqlen"]["xs"], "seqlen", all_data["seqlen"]["results"],
         "Sequence Length (N)", "Latency (ms)", "(a) Latency vs Sequence Length")

    # (b) Latency vs Sparsity (NNZ_S)
    ax1 = axes[0, 1]
    fill(ax1, all_data["sparsity"]["xs"], "sparsity", all_data["sparsity"]["results"],
         "Slash Blocks (NNZ_S)", "Latency (ms)", "(b) Latency vs Slash Sparsity")

    # (c) Latency vs Batch Size
    ax2 = axes[1, 0]
    fill(ax2, all_data["batch"]["xs"], "batch", all_data["batch"]["results"],
         "Batch Size (B)", "Latency (ms)", "(c) Latency vs Batch Size")

    # (d) Latency vs Head Dimension (bar chart)
    ax3 = axes[1, 1]
    Ds = all_data["headdim"]["xs"]
    d_results = all_data["headdim"]["results"]
    x = range(len(Ds))
    width = 0.25
    for i, impl in enumerate(impls):
        ys = [d[impl] * 1000 for d in d_results]
        bars = ax3.bar([xi + i * width for xi in x], ys, width, label=impl, color=colors[impl])
        for bar, v in zip(bars, ys):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax3.set_xticks([xi + width for xi in x])
    ax3.set_xticklabels([f"D={d}" for d in Ds])
    ax3.set_xlabel("Head Dimension")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title("(d) Latency vs Head Dimension", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = OUT_DIR / "benchmark_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {save_path}")
    plt.close(fig)

    # --- Individual speedup bar chart ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Speedup vs Naive PyTorch Implementation", fontsize=14)

    # Seqlen speedup
    ax = axes2[0]
    Ns = all_data["seqlen"]["xs"]
    r = all_data["seqlen"]["results"]
    cuda_speedup = [r["Naive"] / r["CUDA"] for r in r]
    triton_speedup = [r["Naive"] / r["Triton"] for r in r]
    x = range(len(Ns))
    w = 0.35
    ax.bar([xi - w / 2 for xi in x], cuda_speedup, w, label="CUDA", color=colors["CUDA"])
    ax.bar([xi + w / 2 for xi in x], triton_speedup, w, label="Triton", color=colors["Triton"])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel("Speedup over Naive")
    ax.set_title("Speedup vs Sequence Length")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Sparsity speedup
    ax = axes2[1]
    NSs = all_data["sparsity"]["xs"]
    r = all_data["sparsity"]["results"]
    cuda_speedup = [r["Naive"] / r["CUDA"] for r in r]
    triton_speedup = [r["Naive"] / r["Triton"] for r in r]
    x = range(len(NSs))
    ax.bar([xi - w / 2 for xi in x], cuda_speedup, w, label="CUDA", color=colors["CUDA"])
    ax.bar([xi + w / 2 for xi in x], triton_speedup, w, label="Triton", color=colors["Triton"])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ns) for ns in NSs])
    ax.set_xlabel("Slash Blocks (NNZ_S)")
    ax.set_ylabel("Speedup over Naive")
    ax.set_title("Speedup vs Slash Sparsity")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path2 = OUT_DIR / "benchmark_speedup.png"
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"[Saved] {save_path2}")
    plt.close(fig2)


def main():
    print("=" * 60)
    print("Mixed Sparse Attention - Comprehensive Benchmark")
    print("=" * 60)

    all_data = {}

    print("\n[1/4] Sweeping Sequence Length ...")
    Ns, results = run_sweep_seqlen()
    all_data["seqlen"] = {"xs": Ns, "results": results}

    print("\n[2/4] Sweeping Slash Sparsity (NNZ_S) ...")
    NSs, results = run_sweep_sparsity()
    all_data["sparsity"] = {"xs": NSs, "results": results}

    print("\n[3/4] Sweeping Batch Size ...")
    Bs, results = run_sweep_batch()
    all_data["batch"] = {"xs": Bs, "results": results}

    print("\n[4/4] Comparing Head Dimensions ...")
    Ds, results = run_sweep_headdim()
    all_data["headdim"] = {"xs": Ds, "results": results}

    print("\n" + "=" * 60)
    print("Generating plots ...")
    plot_results(all_data)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
