#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom Add";
    m.def("add", &add_cuda, "Add two tensors");
}
