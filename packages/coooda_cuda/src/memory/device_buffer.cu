#include <coooda_cuda/memory/device_buffer.cuh>

#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>

#include <cuda_runtime.h>

#include <string>
#include <utility>

namespace coooda_cuda::memory {

namespace {

std::size_t byte_count(
    std::size_t size
) {
    return size * sizeof(float);
}

void require_matching_count(
    std::size_t actual,
    std::size_t expected,
    const char *operation
) {
    if (actual != expected) {
        coooda_core::fail(std::string(operation) + " count does not match device buffer size");
    }
}

} // namespace

DeviceBuffer::DeviceBuffer(
    std::size_t size
)
    : size_(size) {
    if (size_ == 0) {
        return;
    }

    COODA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), bytes()));
}

DeviceBuffer::DeviceBuffer(
    DeviceBuffer &&other
) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(
    DeviceBuffer &&other
) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }

        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }

    return *this;
}

DeviceBuffer::~DeviceBuffer() noexcept {
    if (data_ != nullptr) {
        (void)cudaFree(data_);
    }
}

std::size_t DeviceBuffer::size() const noexcept { return size_; }

std::size_t DeviceBuffer::bytes() const noexcept { return byte_count(size_); }

bool DeviceBuffer::empty() const noexcept { return size_ == 0; }

float *DeviceBuffer::data() const noexcept { return data_; }

void DeviceBuffer::copy_from_host(
    const float *host_values,
    std::size_t count
) {
    require_matching_count(count, size_, "copy_from_host");
    if (count == 0) {
        return;
    }
    if (host_values == nullptr) {
        coooda_core::fail("copy_from_host requires a non-null host pointer");
    }

    COODA_CUDA_CHECK(cudaMemcpy(data_, host_values, bytes(), cudaMemcpyHostToDevice));
}

void DeviceBuffer::copy_to_host(
    float *host_values,
    std::size_t count
) const {
    require_matching_count(count, size_, "copy_to_host");
    if (count == 0) {
        return;
    }
    if (host_values == nullptr) {
        coooda_core::fail("copy_to_host requires a non-null host pointer");
    }

    COODA_CUDA_CHECK(cudaMemcpy(host_values, data_, bytes(), cudaMemcpyDeviceToHost));
}

std::vector<float> DeviceBuffer::copy_to_host() const {
    std::vector<float> host_values(size_, 0.0f);
    copy_to_host(host_values.data(), host_values.size());
    return host_values;
}

DeviceBuffer DeviceBuffer::from_host(
    const std::vector<float> &host_values
) {
    DeviceBuffer buffer(host_values.size());
    buffer.copy_from_host(host_values.data(), host_values.size());
    return buffer;
}

} // namespace coooda_cuda::memory
