#include <coooda_cuda/nn/loss.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxReductionBlocks = 128;
constexpr int kSoftmaxPlain = 0;
constexpr int kSoftmaxLog = 1;

void require_non_empty_logits(std::size_t size, const char *operation) {
    if (size == 0) {
        coooda_core::fail(std::string(operation) + " requires non-empty logits");
    }
}

void require_target_in_range(std::size_t target_index, std::size_t class_count, const char *operation) {
    if (target_index >= class_count) {
        coooda_core::fail(std::string(operation) + " target index is out of range");
    }
}

void require_batch_shape(
    std::size_t logits_size,
    std::size_t target_count,
    std::size_t class_count,
    const char *operation
) {
    if (target_count == 0) {
        coooda_core::fail(std::string(operation) + " requires at least one target");
    }
    if (class_count == 0) {
        coooda_core::fail(std::string(operation) + " requires non-zero class count");
    }
    if (target_count > (std::numeric_limits<std::size_t>::max() / class_count)) {
        coooda_core::fail(std::string(operation) + " batch shape is too large");
    }
    if (logits_size != target_count * class_count) {
        coooda_core::fail(std::string(operation) + " logits size must equal target count times class count");
    }
}

int checked_count(std::size_t size, const char *operation) {
    if (size > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " input is too large for the CUDA loss kernel");
    }
    return static_cast<int>(size);
}

int reduction_grid_size(int n) {
    const int blocks = (n + kBlockSize - 1) / kBlockSize;
    return blocks < kMaxReductionBlocks ? blocks : kMaxReductionBlocks;
}

std::vector<int> checked_targets(
    const std::vector<std::size_t> &targets,
    std::size_t class_count,
    const char *operation
) {
    std::vector<int> out(targets.size(), 0);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        require_target_in_range(targets[i], class_count, operation);
        out[i] = checked_count(targets[i], operation);
    }
    return out;
}

class DeviceIntBuffer {
public:
    DeviceIntBuffer() = default;

    explicit DeviceIntBuffer(std::size_t size)
        : size_(size) {
        if (size_ == 0) {
            return;
        }

        COODA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(int)));
    }

    DeviceIntBuffer(const DeviceIntBuffer &) = delete;
    DeviceIntBuffer &operator=(const DeviceIntBuffer &) = delete;

    DeviceIntBuffer(DeviceIntBuffer &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    ~DeviceIntBuffer() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    int *data() const noexcept { return data_; }

    void copy_from_host(const std::vector<int> &host_values) {
        if (host_values.size() != size_) {
            coooda_core::fail("DeviceIntBuffer copy_from_host count does not match buffer size");
        }
        if (size_ == 0) {
            return;
        }

        COODA_CUDA_CHECK(cudaMemcpy(data_, host_values.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    }

    static DeviceIntBuffer from_host(const std::vector<int> &host_values) {
        DeviceIntBuffer buffer(host_values.size());
        buffer.copy_from_host(host_values);
        return buffer;
    }

private:
    int *data_ = nullptr;
    std::size_t size_ = 0;
};

__device__ float reduce_row_max(const float *values, int count) {
    __shared__ float partials[kBlockSize];

    float best = -INFINITY;
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        best = fmaxf(best, values[idx]);
    }

    partials[threadIdx.x] = best;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            partials[threadIdx.x] = fmaxf(partials[threadIdx.x], partials[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    return partials[0];
}

__device__ float reduce_shifted_exp_sum(const float *values, int count, float max_value) {
    __shared__ float partials[kBlockSize];

    float total = 0.0f;
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        total += expf(values[idx] - max_value);
    }

    partials[threadIdx.x] = total;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            partials[threadIdx.x] += partials[threadIdx.x + offset];
        }
        __syncthreads();
    }

    return partials[0];
}

__global__ void softmax_kernel(const float *logits, float *out, int n, int op) {
    const float max_logit = reduce_row_max(logits, n);
    const float sum = reduce_shifted_exp_sum(logits, n, max_logit);
    const float log_sum = logf(sum);

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        if (op == kSoftmaxLog) {
            out[idx] = logits[idx] - max_logit - log_sum;
        } else {
            out[idx] = expf(logits[idx] - max_logit) / sum;
        }
    }
}

__global__ void cross_entropy_kernel(const float *logits, float *out, int n, int target_index) {
    const float max_logit = reduce_row_max(logits, n);
    const float sum = reduce_shifted_exp_sum(logits, n, max_logit);

    if (threadIdx.x == 0) {
        out[0] = max_logit + logf(sum) - logits[target_index];
    }
}

__global__ void cross_entropy_rows_kernel(
    const float *logits,
    const int *targets,
    float *losses,
    int class_count
) {
    const int row = blockIdx.x;
    const float *row_logits = logits + (row * class_count);
    const int target_index = targets[row];
    const float max_logit = reduce_row_max(row_logits, class_count);
    const float sum = reduce_shifted_exp_sum(row_logits, class_count, max_logit);

    if (threadIdx.x == 0) {
        losses[row] = max_logit + logf(sum) - row_logits[target_index];
    }
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

std::vector<float> run_softmax(const std::vector<float> &logits, int op, const char *operation) {
    require_non_empty_logits(logits.size(), operation);
    const int n = checked_count(logits.size(), operation);

    coooda_cuda::memory::DeviceBuffer device_logits = coooda_cuda::memory::DeviceBuffer::from_host(logits);
    coooda_cuda::memory::DeviceBuffer device_out(logits.size());

    softmax_kernel<<<1, kBlockSize>>>(device_logits.data(), device_out.data(), n, op);
    COODA_CUDA_CHECK_LAST(operation);
    return device_out.copy_to_host();
}

float reduce_device_sum(
    coooda_cuda::memory::DeviceBuffer current,
    int current_count,
    const char *operation
) {
    while (current_count > 1) {
        const int next_count = reduction_grid_size(current_count);
        coooda_cuda::memory::DeviceBuffer next(static_cast<std::size_t>(next_count));

        sum_blocks_kernel<<<next_count, kBlockSize>>>(current.data(), next.data(), current_count);
        COODA_CUDA_CHECK_LAST(operation);

        current = std::move(next);
        current_count = next_count;
    }

    return current.copy_to_host()[0];
}

} // namespace

namespace coooda_cuda::nn {

std::vector<float> softmax_baseline(const std::vector<float> &logits) {
    return run_softmax(logits, kSoftmaxPlain, "softmax_baseline");
}

std::vector<float> log_softmax_baseline(const std::vector<float> &logits) {
    return run_softmax(logits, kSoftmaxLog, "log_softmax_baseline");
}

float cross_entropy_loss_baseline(
    const std::vector<float> &logits,
    std::size_t target_index
) {
    require_non_empty_logits(logits.size(), "cross_entropy_loss_baseline");
    require_target_in_range(target_index, logits.size(), "cross_entropy_loss_baseline");

    const int n = checked_count(logits.size(), "cross_entropy_loss_baseline");
    const int target = checked_count(target_index, "cross_entropy_loss_baseline");
    coooda_cuda::memory::DeviceBuffer device_logits = coooda_cuda::memory::DeviceBuffer::from_host(logits);
    coooda_cuda::memory::DeviceBuffer device_out(1);

    cross_entropy_kernel<<<1, kBlockSize>>>(device_logits.data(), device_out.data(), n, target);
    COODA_CUDA_CHECK_LAST("cross_entropy_loss_baseline");
    return device_out.copy_to_host()[0];
}

float mean_cross_entropy_loss_baseline(
    const std::vector<float> &logits,
    const std::vector<std::size_t> &targets,
    std::size_t class_count
) {
    require_batch_shape(logits.size(), targets.size(), class_count, "mean_cross_entropy_loss_baseline");

    const int batch_size = checked_count(targets.size(), "mean_cross_entropy_loss_baseline");
    const int classes = checked_count(class_count, "mean_cross_entropy_loss_baseline");
    const std::vector<int> target_indices =
        checked_targets(targets, class_count, "mean_cross_entropy_loss_baseline");

    coooda_cuda::memory::DeviceBuffer device_logits = coooda_cuda::memory::DeviceBuffer::from_host(logits);
    DeviceIntBuffer device_targets = DeviceIntBuffer::from_host(target_indices);
    coooda_cuda::memory::DeviceBuffer device_losses(targets.size());

    cross_entropy_rows_kernel<<<batch_size, kBlockSize>>>(
        device_logits.data(),
        device_targets.data(),
        device_losses.data(),
        classes
    );
    COODA_CUDA_CHECK_LAST("mean_cross_entropy_loss_baseline");

    const std::vector<float> losses = device_losses.copy_to_host();
    float total = 0.0f;
    for (float loss : losses) {
        total += loss;
    }
    return total / static_cast<float>(losses.size());
}

float mean_cross_entropy_loss_device_reduce(
    const std::vector<float> &logits,
    const std::vector<std::size_t> &targets,
    std::size_t class_count
) {
    require_batch_shape(logits.size(), targets.size(), class_count, "mean_cross_entropy_loss_device_reduce");

    const int batch_size = checked_count(targets.size(), "mean_cross_entropy_loss_device_reduce");
    const int classes = checked_count(class_count, "mean_cross_entropy_loss_device_reduce");
    const std::vector<int> target_indices =
        checked_targets(targets, class_count, "mean_cross_entropy_loss_device_reduce");

    coooda_cuda::memory::DeviceBuffer device_logits = coooda_cuda::memory::DeviceBuffer::from_host(logits);
    DeviceIntBuffer device_targets = DeviceIntBuffer::from_host(target_indices);
    coooda_cuda::memory::DeviceBuffer device_losses(targets.size());

    cross_entropy_rows_kernel<<<batch_size, kBlockSize>>>(
        device_logits.data(),
        device_targets.data(),
        device_losses.data(),
        classes
    );
    COODA_CUDA_CHECK_LAST("mean_cross_entropy_loss_device_reduce");

    const float total = reduce_device_sum(
        std::move(device_losses),
        batch_size,
        "mean_cross_entropy_loss_device_reduce sum_blocks_kernel"
    );
    return total / static_cast<float>(targets.size());
}

} // namespace coooda_cuda::nn
