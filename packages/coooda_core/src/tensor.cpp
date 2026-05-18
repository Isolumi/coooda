#include <coooda_core/tensor.hpp>

#include <numeric>
#include <sstream>

namespace coooda_core {

HostBuffer::HostBuffer(
    std::size_t size
)
    : values_(size, 0.0f) {}

std::size_t HostBuffer::size() const noexcept { return values_.size(); }

bool HostBuffer::empty() const noexcept { return values_.empty(); }

float *HostBuffer::data() noexcept { return values_.data(); }

const float *HostBuffer::data() const noexcept { return values_.data(); }

float &HostBuffer::operator[](
    std::size_t index
) noexcept {
    return values_[index];
}

const float &HostBuffer::operator[](
    std::size_t index
) const noexcept {
    return values_[index];
}

std::size_t numel(
    const Shape &shape
) {
    if (shape.dims.empty()) {
        return 0;
    }

    return std::accumulate(
        shape.dims.begin(),
        shape.dims.end(),
        static_cast<std::size_t>(1),
        [](std::size_t acc, std::size_t dim) { return acc * dim; }
    );
}

std::string to_string(
    const Shape &shape
) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.dims.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        out << shape.dims[i];
    }
    out << "]";
    return out.str();
}

TensorView::TensorView(
    const Shape &shape,
    float *data
)
    : shape_(shape), data_(data) {}

const Shape &TensorView::shape() const noexcept { return shape_; }

std::size_t TensorView::size() const { return numel(shape_); }

bool TensorView::empty() const { return size() == 0; }

float *TensorView::data() noexcept { return data_; }

const float *TensorView::data() const noexcept { return data_; }

float &TensorView::operator[](
    std::size_t index
) noexcept {
    return data_[index];
}

const float &TensorView::operator[](
    std::size_t index
) const noexcept {
    return data_[index];
}

Tensor::Tensor(
    const Shape &shape
)
    : shape_(shape), storage_(numel(shape)) {}

const Shape &Tensor::shape() const noexcept { return shape_; }

std::size_t Tensor::size() const { return storage_.size(); }

bool Tensor::empty() const { return storage_.empty(); }

float *Tensor::data() noexcept { return storage_.data(); }

const float *Tensor::data() const noexcept { return storage_.data(); }

float &Tensor::operator[](
    std::size_t index
) noexcept {
    return storage_[index];
}

const float &Tensor::operator[](
    std::size_t index
) const noexcept {
    return storage_[index];
}

TensorView Tensor::view() noexcept {
    return TensorView(shape_, data());
}

} // namespace coooda_core
