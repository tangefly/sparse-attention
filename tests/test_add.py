import pytest
import torch
import sparse_attn

@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("size", [1, 10, 255, 256, 257, 1024, 4096, 100000])
def test_cuda_add(size, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")

    device = "cuda"

    a = torch.randn(size, device=device, dtype=dtype)
    b = torch.randn(size, device=device, dtype=dtype)

    out = sparse_attn.cadd(a, b)
    ref = a + b

    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
