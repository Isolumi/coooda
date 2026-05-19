#include <coooda_cuda/ops/matmul.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 16;

void require_matrix(const coooda_core::Tensor &tensor, const char *name) {
    if (tensor.shape().dims.size() != 2) {
        coooda_core::fail(std::string("matmul_baseline requires ") + name + " to be rank 2");
    }
}

std::size_t rows(const coooda_core::Tensor &tensor) {
    return tensor.shape().dims[0];
}

std::size_t cols(const coooda_core::Tensor &tensor) {
    return tensor.shape().dims[1];
}

int checked_dimension(std::size_t value, const char *operation) {
    if (value > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " dimension is too large for the baseline kernel");
    }
    return static_cast<int>(value);
}

int checked_numel(const coooda_core::Tensor &tensor, const char *operation) {
    if (tensor.size() > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " tensor is too large for the baseline kernel");
    }
    return static_cast<int>(tensor.size());
}

std::vector<float> tensor_values(const coooda_core::Tensor &tensor) {
    std::vector<float> values(tensor.size(), 0.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = tensor[i];
    }
    return values;
}

__global__ void matmul_kernel(
    const float *a,
    const float *b,
    float *out,
    int m,
    int k,
    int n
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float total = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        total += a[(row * k) + inner] * b[(inner * n) + col];
    }

    out[(row * n) + col] = total;
}

} // namespace

namespace coooda_cuda::ops {

coooda_core::Tensor matmul_baseline(
    const coooda_core::Tensor &a,
    const coooda_core::Tensor &b
) {
    require_matrix(a, "left input");
    require_matrix(b, "right input");

    const std::size_t m_size = rows(a);
    const std::size_t k_size = cols(a);
    const std::size_t b_rows = rows(b);
    const std::size_t n_size = cols(b);

    if (k_size != b_rows) {
        coooda_core::fail("matmul_baseline requires left columns to equal right rows");
    }

    coooda_core::Tensor output({{m_size, n_size}});
    if (output.empty()) {
        return output;
    }

    (void)checked_numel(a, "matmul_baseline");
    (void)checked_numel(b, "matmul_baseline");
    (void)checked_numel(output, "matmul_baseline");
    const int m = checked_dimension(m_size, "matmul_baseline");
    const int k = checked_dimension(k_size, "matmul_baseline");
    const int n = checked_dimension(n_size, "matmul_baseline");

    coooda_cuda::memory::DeviceBuffer device_a =
        coooda_cuda::memory::DeviceBuffer::from_host(tensor_values(a));
    coooda_cuda::memory::DeviceBuffer device_b =
        coooda_cuda::memory::DeviceBuffer::from_host(tensor_values(b));
    coooda_cuda::memory::DeviceBuffer device_out(output.size());

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    matmul_kernel<<<grid, block>>>(device_a.data(), device_b.data(), device_out.data(), m, k, n);
    COODA_CUDA_CHECK_LAST("matmul_kernel");

    device_out.copy_to_host(output.data(), output.size());
    return output;
}

} // namespace coooda_cuda::ops
