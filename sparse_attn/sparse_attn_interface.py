import torch
from sparse_attn_cuda import add

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
