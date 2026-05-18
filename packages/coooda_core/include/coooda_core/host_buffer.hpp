#pragma once

#include <cstddef>
#include <vector>

namespace coooda_core {
class HostBuffer {
  public:
    HostBuffer() = default;
    explicit HostBuffer(std::size_t size);

    std::size_t size() const noexcept;
    bool empty() const noexcept;

    float *data() noexcept;
    const float *data() const noexcept;

    float &operator[](std::size_t index) noexcept;
    const float &operator[](std::size_t index) const noexcept;

  private:
    std::vector<float> values_;
};
} // namespace coooda_core