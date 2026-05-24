#include <coooda_cuda/nn/embedding.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxGridStrideBlocks = 256;

std::size_t require_embedding_table_shape(
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim,
    const char *operation
) {
    if (embedding_dim == 0) {
        coooda_core::fail(std::string(operation) + " requires non-zero embedding dim");
    }
    if (embedding_table.size() % embedding_dim != 0) {
        coooda_core::fail(std::string(operation) + " embedding table size must be divisible by embedding dim");
    }
    return embedding_table.size() / embedding_dim;
}

int checked_count(std::size_t size, const char *operation) {
    if (size > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail(std::string(operation) + " input is too large for the CUDA embedding kernel");
    }
    return static_cast<int>(size);
}

std::vector<int> checked_ids(
    const std::vector<std::size_t> &ids,
    std::size_t row_count,
    const char *operation
) {
    std::vector<int> out(ids.size(), 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= row_count) {
            coooda_core::fail(std::string(operation) + " id is out of range");
        }
        out[i] = checked_count(ids[i], operation);
    }
    return out;
}

int grid_size_for_elements(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

int grid_stride_grid_size(int n) {
    const int one_thread_blocks = grid_size_for_elements(n);
    return one_thread_blocks < kMaxGridStrideBlocks ? one_thread_blocks : kMaxGridStrideBlocks;
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

__global__ void embedding_lookup_kernel(
    const int *ids,
    const float *embedding_table,
    float *out,
    int token_count,
    int embedding_dim
) {
    const int total = token_count * embedding_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int token_index = idx / embedding_dim;
    const int dim = idx % embedding_dim;
    const int table_row = ids[token_index];
    out[idx] = embedding_table[(table_row * embedding_dim) + dim];
}

__global__ void token_position_embedding_kernel(
    const int *token_ids,
    const float *token_embedding_table,
    const float *position_embedding_table,
    float *out,
    int token_count,
    int embedding_dim
) {
    const int total = token_count * embedding_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int token_index = idx / embedding_dim;
    const int dim = idx % embedding_dim;
    const int token_row = token_ids[token_index];
    out[idx] =
        token_embedding_table[(token_row * embedding_dim) + dim] +
        position_embedding_table[(token_index * embedding_dim) + dim];
}

__global__ void embedding_lookup_grid_stride_kernel(
    const int *ids,
    const float *embedding_table,
    float *out,
    int token_count,
    int embedding_dim
) {
    const int total = token_count * embedding_dim;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int token_index = idx / embedding_dim;
        const int dim = idx % embedding_dim;
        const int table_row = ids[token_index];
        out[idx] = embedding_table[(table_row * embedding_dim) + dim];
    }
}

__global__ void token_position_embedding_grid_stride_kernel(
    const int *token_ids,
    const float *token_embedding_table,
    const float *position_embedding_table,
    float *out,
    int token_count,
    int embedding_dim
) {
    const int total = token_count * embedding_dim;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int token_index = idx / embedding_dim;
        const int dim = idx % embedding_dim;
        const int token_row = token_ids[token_index];
        out[idx] =
            token_embedding_table[(token_row * embedding_dim) + dim] +
            position_embedding_table[(token_index * embedding_dim) + dim];
    }
}

} // namespace

namespace coooda_cuda::nn {

std::vector<float> embedding_lookup_baseline(
    const std::vector<std::size_t> &ids,
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim
) {
    const std::size_t row_count =
        require_embedding_table_shape(embedding_table, embedding_dim, "embedding_lookup_baseline");
    const std::vector<int> host_ids = checked_ids(ids, row_count, "embedding_lookup_baseline");
    if (ids.empty()) {
        return {};
    }

    const int token_count = checked_count(ids.size(), "embedding_lookup_baseline");
    const int dim = checked_count(embedding_dim, "embedding_lookup_baseline");
    const int total = checked_count(ids.size() * embedding_dim, "embedding_lookup_baseline");
    DeviceIntBuffer device_ids = DeviceIntBuffer::from_host(host_ids);
    coooda_cuda::memory::DeviceBuffer device_table = coooda_cuda::memory::DeviceBuffer::from_host(embedding_table);
    coooda_cuda::memory::DeviceBuffer device_out(ids.size() * embedding_dim);

    embedding_lookup_kernel<<<grid_size_for_elements(total), kBlockSize>>>(
        device_ids.data(),
        device_table.data(),
        device_out.data(),
        token_count,
        dim
    );
    COODA_CUDA_CHECK_LAST("embedding_lookup_baseline");
    return device_out.copy_to_host();
}

std::vector<float> token_position_embedding_baseline(
    const std::vector<std::size_t> &token_ids,
    const std::vector<float> &token_embedding_table,
    const std::vector<float> &position_embedding_table,
    std::size_t embedding_dim
) {
    const std::size_t vocab_size =
        require_embedding_table_shape(token_embedding_table, embedding_dim, "token_position_embedding_baseline");
    const std::size_t position_count =
        require_embedding_table_shape(position_embedding_table, embedding_dim, "token_position_embedding_baseline");
    const std::vector<int> host_ids = checked_ids(token_ids, vocab_size, "token_position_embedding_baseline");
    if (token_ids.size() > position_count) {
        coooda_core::fail("token_position_embedding_baseline position table is too short");
    }
    if (token_ids.empty()) {
        return {};
    }

    const int token_count = checked_count(token_ids.size(), "token_position_embedding_baseline");
    const int dim = checked_count(embedding_dim, "token_position_embedding_baseline");
    const int total = checked_count(token_ids.size() * embedding_dim, "token_position_embedding_baseline");
    DeviceIntBuffer device_ids = DeviceIntBuffer::from_host(host_ids);
    coooda_cuda::memory::DeviceBuffer device_token_table =
        coooda_cuda::memory::DeviceBuffer::from_host(token_embedding_table);
    coooda_cuda::memory::DeviceBuffer device_position_table =
        coooda_cuda::memory::DeviceBuffer::from_host(position_embedding_table);
    coooda_cuda::memory::DeviceBuffer device_out(token_ids.size() * embedding_dim);

    token_position_embedding_kernel<<<grid_size_for_elements(total), kBlockSize>>>(
        device_ids.data(),
        device_token_table.data(),
        device_position_table.data(),
        device_out.data(),
        token_count,
        dim
    );
    COODA_CUDA_CHECK_LAST("token_position_embedding_baseline");
    return device_out.copy_to_host();
}

std::vector<float> embedding_lookup_grid_stride(
    const std::vector<std::size_t> &ids,
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim
) {
    const std::size_t row_count =
        require_embedding_table_shape(embedding_table, embedding_dim, "embedding_lookup_grid_stride");
    const std::vector<int> host_ids = checked_ids(ids, row_count, "embedding_lookup_grid_stride");
    if (ids.empty()) {
        return {};
    }

    const int token_count = checked_count(ids.size(), "embedding_lookup_grid_stride");
    const int dim = checked_count(embedding_dim, "embedding_lookup_grid_stride");
    const int total = checked_count(ids.size() * embedding_dim, "embedding_lookup_grid_stride");
    DeviceIntBuffer device_ids = DeviceIntBuffer::from_host(host_ids);
    coooda_cuda::memory::DeviceBuffer device_table = coooda_cuda::memory::DeviceBuffer::from_host(embedding_table);
    coooda_cuda::memory::DeviceBuffer device_out(ids.size() * embedding_dim);

    embedding_lookup_grid_stride_kernel<<<grid_stride_grid_size(total), kBlockSize>>>(
        device_ids.data(),
        device_table.data(),
        device_out.data(),
        token_count,
        dim
    );
    COODA_CUDA_CHECK_LAST("embedding_lookup_grid_stride");
    return device_out.copy_to_host();
}

std::vector<float> token_position_embedding_grid_stride(
    const std::vector<std::size_t> &token_ids,
    const std::vector<float> &token_embedding_table,
    const std::vector<float> &position_embedding_table,
    std::size_t embedding_dim
) {
    const std::size_t vocab_size =
        require_embedding_table_shape(token_embedding_table, embedding_dim, "token_position_embedding_grid_stride");
    const std::size_t position_count =
        require_embedding_table_shape(position_embedding_table, embedding_dim, "token_position_embedding_grid_stride");
    const std::vector<int> host_ids = checked_ids(token_ids, vocab_size, "token_position_embedding_grid_stride");
    if (token_ids.size() > position_count) {
        coooda_core::fail("token_position_embedding_grid_stride position table is too short");
    }
    if (token_ids.empty()) {
        return {};
    }

    const int token_count = checked_count(token_ids.size(), "token_position_embedding_grid_stride");
    const int dim = checked_count(embedding_dim, "token_position_embedding_grid_stride");
    const int total = checked_count(token_ids.size() * embedding_dim, "token_position_embedding_grid_stride");
    DeviceIntBuffer device_ids = DeviceIntBuffer::from_host(host_ids);
    coooda_cuda::memory::DeviceBuffer device_token_table =
        coooda_cuda::memory::DeviceBuffer::from_host(token_embedding_table);
    coooda_cuda::memory::DeviceBuffer device_position_table =
        coooda_cuda::memory::DeviceBuffer::from_host(position_embedding_table);
    coooda_cuda::memory::DeviceBuffer device_out(token_ids.size() * embedding_dim);

    token_position_embedding_grid_stride_kernel<<<grid_stride_grid_size(total), kBlockSize>>>(
        device_ids.data(),
        device_token_table.data(),
        device_position_table.data(),
        device_out.data(),
        token_count,
        dim
    );
    COODA_CUDA_CHECK_LAST("token_position_embedding_grid_stride");
    return device_out.copy_to_host();
}

} // namespace coooda_cuda::nn
