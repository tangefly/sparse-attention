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

void launch_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(a, b, out, n);
}
