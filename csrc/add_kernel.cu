#include <torch/extension.h>

__global__ void add_kernel(
    const float* a,
    const float* b,
    float* out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b)
{
    auto out = torch::empty_like(a);

    int n = a.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}
