#pragma once

#include <coooda_core/host_buffer.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace coooda_core {

struct Shape {
    std::vector<std::size_t> dims;
};

std::size_t numel(const Shape &shape);
std::string to_string(const Shape &shape);

class TensorView {
  public:
    TensorView() = default;
    TensorView(const Shape &shape, float *data);

    const Shape &shape() const noexcept;
    std::size_t size() const;
    bool empty() const;

    float *data() noexcept;
    const float *data() const noexcept;

    float &operator[](std::size_t index) noexcept;
    const float &operator[](std::size_t index) const noexcept;

  private:
    Shape shape_{};
    float *data_ = nullptr;
};

class Tensor {
  public:
    Tensor() = default;
    explicit Tensor(const Shape &shape);

    const Shape &shape() const noexcept;
    std::size_t size() const;
    bool empty() const;

    float *data() noexcept;
    const float *data() const noexcept;

    float &operator[](std::size_t index) noexcept;
    const float &operator[](std::size_t index) const noexcept;

    TensorView view() noexcept;

  private:
    Shape shape_{};
    HostBuffer storage_{};
};

} // namespace coooda_core
