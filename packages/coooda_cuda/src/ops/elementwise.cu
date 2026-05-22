#include <coooda_cuda/ops/elementwise.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr int kUnaryRelu = 0;
constexpr int kUnaryGelu = 1;
constexpr int kBinaryAdd = 0;
constexpr int kBinaryMultiply = 1;
constexpr int kBiasPlain = 0;
constexpr int kBiasRelu = 1;
constexpr int kBiasGelu = 2;
constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
constexpr float kGeluCoeff = 0.044715f;

void require_same_size(
    const std::vector<float> &a,
    const std::vector<float> &b,
    const char *operation
) {
    if (a.size() != b.size()) {
        coooda_core::fail(std::string(operation) + " requires equal input sizes");
    }
}

void require_bias_shape(
    const std::vector<float> &input,
    const std::vector<float> &bias,
    const char *operation
) {
    if (input.empty()) {
        return;
    }
    if (bias.empty()) {
        coooda_core::fail(std::string(operation) + " requires non-empty bias for non-empty input");
    }
    if (input.size() % bias.size() != 0) {
        coooda_core::fail(std::string(operation) + " requires input size to be divisible by bias size");
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

__device__ float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

__device__ float gelu(float value) {
    const float cubic = value * value * value;
    return 0.5f * value * (1.0f + tanhf(kSqrtTwoOverPi * (value + (kGeluCoeff * cubic))));
}

__device__ float apply_bias_op(float value, int op) {
    if (op == kBiasRelu) {
        return relu(value);
    }
    if (op == kBiasGelu) {
        return gelu(value);
    }
    return value;
}

__global__ void unary_kernel(const float *input, float *out, int n, int op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    if (op == kUnaryRelu) {
        out[idx] = relu(input[idx]);
    } else {
        out[idx] = gelu(input[idx]);
    }
}

__global__ void binary_kernel(const float *a, const float *b, float *out, int n, int op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    if (op == kBinaryAdd) {
        out[idx] = a[idx] + b[idx];
    } else {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void add_scalar_kernel(const float *input, float scalar, float *out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = input[idx] + scalar;
    }
}

__global__ void add_bias_kernel(const float *input, const float *bias, float *out, int n, int bias_size, int op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = apply_bias_op(input[idx] + bias[idx % bias_size], op);
    }
}

std::vector<float> run_unary_baseline(const std::vector<float> &input, int op, const char *operation) {
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), operation);
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    unary_kernel<<<one_thread_per_element_grid_size(n), kBlockSize>>>(device_input.data(), device_out.data(), n, op);
    COODA_CUDA_CHECK_LAST(operation);
    return device_out.copy_to_host();
}

std::vector<float> run_binary_baseline(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int op,
    const char *operation
) {
    require_same_size(a, b, operation);
    if (a.empty()) {
        return {};
    }

    const int n = checked_element_count(a.size(), operation);
    coooda_cuda::memory::DeviceBuffer device_a = coooda_cuda::memory::DeviceBuffer::from_host(a);
    coooda_cuda::memory::DeviceBuffer device_b = coooda_cuda::memory::DeviceBuffer::from_host(b);
    coooda_cuda::memory::DeviceBuffer device_out(a.size());

    binary_kernel<<<one_thread_per_element_grid_size(n), kBlockSize>>>(
        device_a.data(),
        device_b.data(),
        device_out.data(),
        n,
        op
    );
    COODA_CUDA_CHECK_LAST(operation);
    return device_out.copy_to_host();
}

std::vector<float> run_bias_baseline(
    const std::vector<float> &input,
    const std::vector<float> &bias,
    int op,
    const char *operation
) {
    require_bias_shape(input, bias, operation);
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), operation);
    const int bias_size = checked_element_count(bias.size(), operation);
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_bias = coooda_cuda::memory::DeviceBuffer::from_host(bias);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    add_bias_kernel<<<one_thread_per_element_grid_size(n), kBlockSize>>>(
        device_input.data(),
        device_bias.data(),
        device_out.data(),
        n,
        bias_size,
        op
    );
    COODA_CUDA_CHECK_LAST(operation);
    return device_out.copy_to_host();
}

} // namespace

namespace coooda_cuda::ops {

std::vector<float> unary_relu_baseline(const std::vector<float> &input) {
    return run_unary_baseline(input, kUnaryRelu, "unary_relu_baseline");
}

std::vector<float> unary_gelu_baseline(const std::vector<float> &input) {
    return run_unary_baseline(input, kUnaryGelu, "unary_gelu_baseline");
}

std::vector<float> binary_add_baseline(const std::vector<float> &a, const std::vector<float> &b) {
    return run_binary_baseline(a, b, kBinaryAdd, "binary_add_baseline");
}

std::vector<float> binary_multiply_baseline(const std::vector<float> &a, const std::vector<float> &b) {
    return run_binary_baseline(a, b, kBinaryMultiply, "binary_multiply_baseline");
}

std::vector<float> add_scalar_baseline(const std::vector<float> &input, float scalar) {
    if (input.empty()) {
        return {};
    }

    const int n = checked_element_count(input.size(), "add_scalar_baseline");
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_out(input.size());

    add_scalar_kernel<<<one_thread_per_element_grid_size(n), kBlockSize>>>(
        device_input.data(),
        scalar,
        device_out.data(),
        n
    );
    COODA_CUDA_CHECK_LAST("add_scalar_baseline");
    return device_out.copy_to_host();
}

std::vector<float> add_bias_baseline(const std::vector<float> &input, const std::vector<float> &bias) {
    return run_bias_baseline(input, bias, kBiasPlain, "add_bias_baseline");
}

std::vector<float> add_bias_relu_baseline(const std::vector<float> &input, const std::vector<float> &bias) {
    return run_bias_baseline(input, bias, kBiasRelu, "add_bias_relu_baseline");
}

std::vector<float> add_bias_gelu_baseline(const std::vector<float> &input, const std::vector<float> &bias) {
    return run_bias_baseline(input, bias, kBiasGelu, "add_bias_gelu_baseline");
}

} // namespace coooda_cuda::ops
