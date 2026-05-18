#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cuda::memory {

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t size);
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
    ~DeviceBuffer() noexcept;

    std::size_t size() const noexcept;
    std::size_t bytes() const noexcept;
    bool empty() const noexcept;

    float *data() const noexcept;

    void copy_from_host(const float *host_values, std::size_t count);
    void copy_to_host(float *host_values, std::size_t count) const;
    std::vector<float> copy_to_host() const;

    static DeviceBuffer from_host(const std::vector<float> &host_values);

private:
    float *data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace coooda_cuda::memory
