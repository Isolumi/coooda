#include <coooda_cuda/ops/vector.cuh>
#include <coooda_cuda/memory/device_buffer.cuh>
#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr int kBinaryAdd = 0;
constexpr int kBinaryMultiply = 1;
constexpr int kUnaryRelu = 0;
constexpr int kUnarySigmoid = 1;
constexpr int kMaxGridStrideBlocks = 128;

void require_same_size(
    const std::vector<float> &a,
    const std::vector<float> &b,
    const char *operation
) {
    if (a.size() != b.size()) {
        coooda_core::fail(std::string(operation) + " requires equal input sizes");
    }
}

int checked_element_count(std::size_t size, const char *operation) {
    if (size > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " input is too large for the baseline kernel");
    }
    return static_cast<int>(size);
}

int one_thread_per_element_grid_size(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

int grid_stride_grid_size(int n) {
    return std::min(one_thread_per_element_grid_size(n), kMaxGridStrideBlocks);
}

__global__ void binary_kernel(const float *a, const float *b, float *out, int n, int op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (op == kBinaryAdd) {
            out[idx] = a[idx] + b[idx];
        } else {
            out[idx] = a[idx] * b[idx];
        }
    }
}

__global__ void binary_grid_stride_kernel(const float *a, const float *b, float *out, int n, int op) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        if (op == kBinaryAdd) {
            out[idx] = a[idx] + b[idx];
        } else {
            out[idx] = a[idx] * b[idx];
        }
    }
}

__global__ void saxpy_kernel(float alpha, const float *x, const float *y, float *out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (alpha * x[idx]) + y[idx];
    }
}

__global__ void saxpy_grid_stride_kernel(float alpha, const float *x, const float *y, float *out, int n) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        out[idx] = (alpha * x[idx]) + y[idx];
    }
}

__global__ void unary_kernel(const float *input, float *out, int n, int op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (op == kUnaryRelu) {
            out[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
        } else {
            out[idx] = 1.0f / (1.0f + expf(-input[idx]));
        }
    }
}

__global__ void unary_grid_stride_kernel(const float *input, float *out, int n, int op) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        if (op == kUnaryRelu) {
            out[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
        } else {
            out[idx] = 1.0f / (1.0f + expf(-input[idx]));
        }
    }
}

} // namespace

namespace coooda_cuda::ops {

std::vector<float> vector_add_baseline(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "vector_add_baseline");

    if (a.empty()) {
        return {};
    }

    const int n = checked_element_count(a.size(), "vector_add_baseline");
    coooda_cuda::memory::DeviceBuffer device_a = coooda_cuda::memory::DeviceBuffer::from_host(a);
    coooda_cuda::memory::DeviceBuffer device_b = coooda_cuda::memory::DeviceBuffer::from_host(b);
    coooda_cuda::memory::DeviceBuffer device_out(a.size());

    const int grid_size = one_thread_per_element_grid_size(n);
    binary_kernel<<<grid_size, kBlockSize>>>(device_a.data(), device_b.data(), device_out.data(), n, kBinaryAdd);
    COODA_CUDA_CHECK_LAST("binary_kernel(vector_add)");

    return device_out.copy_to_host();
}

std::vector<float> saxpy_baseline(float alpha, const std::vector<float> &x, const std::vector<float> &y) {
    require_same_size(x, y, "saxpy_baseline");

    if (x.empty()) {
        return {};
    }

    const int n = checked_element_count(x.size(), "saxpy_baseline");
    coooda_cuda::memory::DeviceBuffer device_x = coooda_cuda::memory::DeviceBuffer::from_host(x);
    coooda_cuda::memory::DeviceBuffer device_y = coooda_cuda::memory::DeviceBuffer::from_host(y);
    coooda_cuda::memory::DeviceBuffer device_out(x.size());

    const int grid_size = one_thread_per_element_grid_size(n);
    saxpy_kernel<<<grid_size, kBlockSize>>>(alpha, device_x.data(), device_y.data(), device_out.data(), n);
    COODA_CUDA_CHECK_LAST("saxpy_kernel");

    return device_out.copy_to_host();
}

std::vector<float> relu_baseline(const std::vector<float> &input) {
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), "relu_baseline");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    const int grid_size = one_thread_per_element_grid_size(n);
    unary_kernel<<<grid_size, kBlockSize>>>(device_input.data(), device_out.data(), n, kUnaryRelu);
    COODA_CUDA_CHECK_LAST("unary_kernel(relu)");

    return device_out.copy_to_host();
}

std::vector<float> sigmoid_baseline(const std::vector<float> &input) {
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), "sigmoid_baseline");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    const int grid_size = one_thread_per_element_grid_size(n);
    unary_kernel<<<grid_size, kBlockSize>>>(device_input.data(), device_out.data(), n, kUnarySigmoid);
    COODA_CUDA_CHECK_LAST("unary_kernel(sigmoid)");

    return device_out.copy_to_host();
}

std::vector<float> elementwise_multiply_baseline(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "elementwise_multiply_baseline");

    if (a.empty()) {
        return {};
    }

    const int n = checked_element_count(a.size(), "elementwise_multiply_baseline");
    coooda_cuda::memory::DeviceBuffer device_a = coooda_cuda::memory::DeviceBuffer::from_host(a);
    coooda_cuda::memory::DeviceBuffer device_b = coooda_cuda::memory::DeviceBuffer::from_host(b);
    coooda_cuda::memory::DeviceBuffer device_out(a.size());

    const int grid_size = one_thread_per_element_grid_size(n);
    binary_kernel<<<grid_size, kBlockSize>>>(
        device_a.data(),
        device_b.data(),
        device_out.data(),
        n,
        kBinaryMultiply
    );
    COODA_CUDA_CHECK_LAST("binary_kernel(elementwise_multiply)");

    return device_out.copy_to_host();
}

std::vector<float> vector_add_grid_stride(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "vector_add_grid_stride");

    if (a.empty()) {
        return {};
    }

    const int n = checked_element_count(a.size(), "vector_add_grid_stride");
    coooda_cuda::memory::DeviceBuffer device_a = coooda_cuda::memory::DeviceBuffer::from_host(a);
    coooda_cuda::memory::DeviceBuffer device_b = coooda_cuda::memory::DeviceBuffer::from_host(b);
    coooda_cuda::memory::DeviceBuffer device_out(a.size());

    binary_grid_stride_kernel<<<grid_stride_grid_size(n), kBlockSize>>>(
        device_a.data(),
        device_b.data(),
        device_out.data(),
        n,
        kBinaryAdd
    );
    COODA_CUDA_CHECK_LAST("binary_grid_stride_kernel(vector_add)");

    return device_out.copy_to_host();
}

std::vector<float> saxpy_grid_stride(float alpha, const std::vector<float> &x, const std::vector<float> &y) {
    require_same_size(x, y, "saxpy_grid_stride");

    if (x.empty()) {
        return {};
    }

    const int n = checked_element_count(x.size(), "saxpy_grid_stride");
    coooda_cuda::memory::DeviceBuffer device_x = coooda_cuda::memory::DeviceBuffer::from_host(x);
    coooda_cuda::memory::DeviceBuffer device_y = coooda_cuda::memory::DeviceBuffer::from_host(y);
    coooda_cuda::memory::DeviceBuffer device_out(x.size());

    saxpy_grid_stride_kernel<<<grid_stride_grid_size(n), kBlockSize>>>(
        alpha,
        device_x.data(),
        device_y.data(),
        device_out.data(),
        n
    );
    COODA_CUDA_CHECK_LAST("saxpy_grid_stride_kernel");

    return device_out.copy_to_host();
}

std::vector<float> relu_grid_stride(const std::vector<float> &input) {
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), "relu_grid_stride");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    unary_grid_stride_kernel<<<grid_stride_grid_size(n), kBlockSize>>>(
        device_input.data(),
        device_out.data(),
        n,
        kUnaryRelu
    );
    COODA_CUDA_CHECK_LAST("unary_grid_stride_kernel(relu)");

    return device_out.copy_to_host();
}

std::vector<float> sigmoid_grid_stride(const std::vector<float> &input) {
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), "sigmoid_grid_stride");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    unary_grid_stride_kernel<<<grid_stride_grid_size(n), kBlockSize>>>(
        device_input.data(),
        device_out.data(),
        n,
        kUnarySigmoid
    );
    COODA_CUDA_CHECK_LAST("unary_grid_stride_kernel(sigmoid)");

    return device_out.copy_to_host();
}

std::vector<float> elementwise_multiply_grid_stride(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "elementwise_multiply_grid_stride");

    if (a.empty()) {
        return {};
    }

    const int n = checked_element_count(a.size(), "elementwise_multiply_grid_stride");
    coooda_cuda::memory::DeviceBuffer device_a = coooda_cuda::memory::DeviceBuffer::from_host(a);
    coooda_cuda::memory::DeviceBuffer device_b = coooda_cuda::memory::DeviceBuffer::from_host(b);
    coooda_cuda::memory::DeviceBuffer device_out(a.size());

    binary_grid_stride_kernel<<<grid_stride_grid_size(n), kBlockSize>>>(
        device_a.data(),
        device_b.data(),
        device_out.data(),
        n,
        kBinaryMultiply
    );
    COODA_CUDA_CHECK_LAST("binary_grid_stride_kernel(elementwise_multiply)");

    return device_out.copy_to_host();
}

} // namespace coooda_cuda::ops
