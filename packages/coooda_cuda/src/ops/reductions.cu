#include <coooda_cuda/ops/reductions.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxReductionBlocks = 128;

int checked_element_count(std::size_t size, const char *operation) {
    if (size > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " input is too large for the baseline kernel");
    }
    return static_cast<int>(size);
}

int reduction_grid_size(int n) {
    return std::min((n + kBlockSize - 1) / kBlockSize, kMaxReductionBlocks);
}

class DeviceIndexBuffer {
public:
    explicit DeviceIndexBuffer(std::size_t size)
        : size_(size) {
        if (size_ == 0) {
            return;
        }

        COODA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(int)));
    }

    DeviceIndexBuffer(const DeviceIndexBuffer &) = delete;
    DeviceIndexBuffer &operator=(const DeviceIndexBuffer &) = delete;

    ~DeviceIndexBuffer() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    int *data() const noexcept { return data_; }

    std::vector<int> copy_to_host() const {
        std::vector<int> host_values(size_, 0);
        if (size_ == 0) {
            return host_values;
        }

        COODA_CUDA_CHECK(cudaMemcpy(host_values.data(), data_, size_ * sizeof(int), cudaMemcpyDeviceToHost));
        return host_values;
    }

private:
    int *data_ = nullptr;
    std::size_t size_ = 0;
};

__host__ __device__ bool is_better_argmax(float candidate_value, int candidate_index, float best_value, int best_index) {
    return candidate_value > best_value || (candidate_value == best_value && candidate_index < best_index);
}

__global__ void sum_blocks_kernel(const float *input, float *partials, int n) {
    __shared__ float values[kBlockSize];

    float total = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        total += input[idx];
    }

    values[threadIdx.x] = total;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            values[threadIdx.x] += values[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = values[0];
    }
}

__global__ void max_blocks_kernel(const float *input, float *partials, int n) {
    __shared__ float values[kBlockSize];

    float best = input[0];
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        if (input[idx] > best) {
            best = input[idx];
        }
    }

    values[threadIdx.x] = best;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            if (values[threadIdx.x + offset] > values[threadIdx.x]) {
                values[threadIdx.x] = values[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = values[0];
    }
}

__global__ void argmax_blocks_kernel(const float *input, float *partial_values, int *partial_indices, int n) {
    __shared__ float values[kBlockSize];
    __shared__ int indices[kBlockSize];

    float best_value = input[0];
    int best_index = 0;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        const float value = input[idx];
        if (is_better_argmax(value, idx, best_value, best_index)) {
            best_value = value;
            best_index = idx;
        }
    }

    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const float candidate_value = values[threadIdx.x + offset];
            const int candidate_index = indices[threadIdx.x + offset];
            if (is_better_argmax(candidate_value, candidate_index, values[threadIdx.x], indices[threadIdx.x])) {
                values[threadIdx.x] = candidate_value;
                indices[threadIdx.x] = candidate_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}

} // namespace

namespace coooda_cuda::ops {

float sum_baseline(const std::vector<float> &input) {
    if (input.empty()) {
        return 0.0f;
    }

    const int n = checked_element_count(input.size(), "sum_baseline");
    const int grid_size = reduction_grid_size(n);
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_partials(static_cast<std::size_t>(grid_size));

    sum_blocks_kernel<<<grid_size, kBlockSize>>>(device_input.data(), device_partials.data(), n);
    COODA_CUDA_CHECK_LAST("sum_blocks_kernel");

    const std::vector<float> partials = device_partials.copy_to_host();
    float total = 0.0f;
    for (float partial : partials) {
        total += partial;
    }
    return total;
}

float max_baseline(const std::vector<float> &input) {
    if (input.empty()) {
        coooda_core::fail("max_baseline requires a non-empty input");
    }

    const int n = checked_element_count(input.size(), "max_baseline");
    const int grid_size = reduction_grid_size(n);
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_partials(static_cast<std::size_t>(grid_size));

    max_blocks_kernel<<<grid_size, kBlockSize>>>(device_input.data(), device_partials.data(), n);
    COODA_CUDA_CHECK_LAST("max_blocks_kernel");

    const std::vector<float> partials = device_partials.copy_to_host();
    float best = partials[0];
    for (std::size_t i = 1; i < partials.size(); ++i) {
        if (partials[i] > best) {
            best = partials[i];
        }
    }
    return best;
}

float mean_baseline(const std::vector<float> &input) {
    if (input.empty()) {
        coooda_core::fail("mean_baseline requires a non-empty input");
    }

    return sum_baseline(input) / static_cast<float>(input.size());
}

std::size_t argmax_baseline(const std::vector<float> &input) {
    if (input.empty()) {
        coooda_core::fail("argmax_baseline requires a non-empty input");
    }

    const int n = checked_element_count(input.size(), "argmax_baseline");
    const int grid_size = reduction_grid_size(n);
    coooda_cuda::memory::DeviceBuffer device_input = coooda_cuda::memory::DeviceBuffer::from_host(input);
    coooda_cuda::memory::DeviceBuffer device_partial_values(static_cast<std::size_t>(grid_size));
    DeviceIndexBuffer device_partial_indices(static_cast<std::size_t>(grid_size));

    argmax_blocks_kernel<<<grid_size, kBlockSize>>>(
        device_input.data(),
        device_partial_values.data(),
        device_partial_indices.data(),
        n
    );
    COODA_CUDA_CHECK_LAST("argmax_blocks_kernel");

    const std::vector<float> partial_values = device_partial_values.copy_to_host();
    const std::vector<int> partial_indices = device_partial_indices.copy_to_host();

    float best_value = partial_values[0];
    int best_index = partial_indices[0];
    for (std::size_t i = 1; i < partial_values.size(); ++i) {
        if (is_better_argmax(partial_values[i], partial_indices[i], best_value, best_index)) {
            best_value = partial_values[i];
            best_index = partial_indices[i];
        }
    }

    return static_cast<std::size_t>(best_index);
}

} // namespace coooda_cuda::ops
