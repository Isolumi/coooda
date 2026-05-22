#include <coooda_cpp/ops/elementwise.hpp>

#include <coooda_core/status.hpp>

#include <cmath>
#include <cstddef>
#include <string>

namespace {

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

float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

float gelu(float value) {
    const float cubic = value * value * value;
    return 0.5f * value * (1.0f + std::tanh(kSqrtTwoOverPi * (value + (kGeluCoeff * cubic))));
}

} // namespace

namespace coooda_cpp::ops {

std::vector<float> unary_relu_reference(const std::vector<float> &input) {
    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = relu(input[i]);
    }
    return out;
}

std::vector<float> unary_gelu_reference(const std::vector<float> &input) {
    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = gelu(input[i]);
    }
    return out;
}

std::vector<float> binary_add_reference(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "binary_add_reference");

    std::vector<float> out(a.size(), 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<float> binary_multiply_reference(const std::vector<float> &a, const std::vector<float> &b) {
    require_same_size(a, b, "binary_multiply_reference");

    std::vector<float> out(a.size(), 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

std::vector<float> add_scalar_reference(const std::vector<float> &input, float scalar) {
    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i] + scalar;
    }
    return out;
}

std::vector<float> add_bias_reference(const std::vector<float> &input, const std::vector<float> &bias) {
    require_bias_shape(input, bias, "add_bias_reference");

    std::vector<float> out(input.size(), 0.0f);
    if (input.empty()) {
        return out;
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i] + bias[i % bias.size()];
    }
    return out;
}

std::vector<float> add_bias_relu_reference(const std::vector<float> &input, const std::vector<float> &bias) {
    require_bias_shape(input, bias, "add_bias_relu_reference");

    std::vector<float> out(input.size(), 0.0f);
    if (input.empty()) {
        return out;
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = relu(input[i] + bias[i % bias.size()]);
    }
    return out;
}

std::vector<float> add_bias_gelu_reference(const std::vector<float> &input, const std::vector<float> &bias) {
    require_bias_shape(input, bias, "add_bias_gelu_reference");

    std::vector<float> out(input.size(), 0.0f);
    if (input.empty()) {
        return out;
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = gelu(input[i] + bias[i % bias.size()]);
    }
    return out;
}

} // namespace coooda_cpp::ops
