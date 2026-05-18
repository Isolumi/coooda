#include <coooda_cpp/ops/vector.hpp>
#include <coooda_core/status.hpp>

#include <cmath>
#include <cstddef>
#include <string>

namespace coooda_cpp::ops {

namespace {

void require_same_size(
    const std::vector<float> &a,
    const std::vector<float> &b,
    const char *operation
) {
    if (a.size() != b.size()) {
        coooda_core::fail(std::string(operation) + " requires equal input sizes");
    }
}

} // namespace

std::vector<float> vector_add_reference(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "vector_add_reference");

    std::vector<float> out(a.size(), 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<float> saxpy_reference(float alpha, const std::vector<float> &x, const std::vector<float> &y) {
    require_same_size(x, y, "saxpy_reference");

    std::vector<float> out(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = (alpha * x[i]) + y[i];
    }
    return out;
}

std::vector<float> relu_reference(const std::vector<float> &input) {
    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
    return out;
}

std::vector<float> sigmoid_reference(const std::vector<float> &input) {
    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    return out;
}

std::vector<float> elementwise_multiply_reference(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "elementwise_multiply_reference");

    std::vector<float> out(a.size(), 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

} // namespace coooda_cpp::ops
