#pragma once

#include <vector>

namespace coooda_cuda::ops {

std::vector<float> unary_relu_baseline(const std::vector<float> &input);
std::vector<float> unary_gelu_baseline(const std::vector<float> &input);

std::vector<float> binary_add_baseline(const std::vector<float> &a, const std::vector<float> &b);
std::vector<float> binary_multiply_baseline(const std::vector<float> &a, const std::vector<float> &b);

std::vector<float> add_scalar_baseline(const std::vector<float> &input, float scalar);
std::vector<float> add_bias_baseline(const std::vector<float> &input, const std::vector<float> &bias);

std::vector<float> add_bias_relu_baseline(const std::vector<float> &input, const std::vector<float> &bias);
std::vector<float> add_bias_gelu_baseline(const std::vector<float> &input, const std::vector<float> &bias);

} // namespace coooda_cuda::ops
