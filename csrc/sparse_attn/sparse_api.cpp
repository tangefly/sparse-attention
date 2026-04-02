#include <torch/extension.h>
#include "add.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);

    int n = a.numel();

    launch_add_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

std::vector<at::Tensor>
mha_fwd_sparse(at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               const at::Tensor &block_count,
               const at::Tensor &block_offset,
               const at::Tensor &column_count,
               const at::Tensor &column_index,
               const std::optional<at::Tensor> &out_,
               const std::optional<at::Tensor> &alibi_slopes_,
               const double p_dropout,
               const double softmax_scale,
               bool is_causal,
               const double softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_) {

    // ========================
    // 1. 基本shape解析
    // ========================
    const auto batch_size = q.size(0);
    const auto seqlen_q   = q.size(1);
    const auto num_heads  = q.size(2);
    const auto head_size  = q.size(3);

    // ========================
    // 2. 构造 out
    // ========================
    at::Tensor out;

    if (out_.has_value()) {
        out = out_.value();
    } else {
        out = at::empty_like(q);
    }

    // ========================
    // 3. 构造 softmax_lse
    // ========================
    // FlashAttention风格： (batch, num_heads, seqlen_q)
    at::Tensor softmax_lse = at::empty(
        {batch_size, num_heads, seqlen_q},
        q.options().dtype(at::kFloat)   // 一般用 float32 提高数值稳定性
    );

    // ========================
    // 4. (这里应调用你的CUDA kernel)
    // ========================
    // launch_sparse_mha_kernel(
    //     q, k, v,
    //     block_count, block_offset,
    //     column_count, column_index,
    //     out,
    //     softmax_lse,
    //     alibi_slopes_,
    //     p_dropout,
    //     softmax_scale,
    //     is_causal,
    //     softcap,
    //     return_softmax,
    //     gen_
    // );

    return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sparse Attention";
    m.def("add", &add_cuda, "Add two tensors");
    m.def("mha_fwd_sparse", &mha_fwd_sparse, "Sparse Attention");
}