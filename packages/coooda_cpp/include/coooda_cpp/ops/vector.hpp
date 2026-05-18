#pragma once

#include <vector>

namespace coooda_cpp::ops {

std::vector<float> vector_add_reference(const std::vector<float> &a, const std::vector<float> &b);
std::vector<float>
saxpy_reference(float alpha, const std::vector<float> &x, const std::vector<float> &y);
std::vector<float> relu_reference(const std::vector<float> &input);
std::vector<float> sigmoid_reference(const std::vector<float> &input);
std::vector<float>
elementwise_multiply_reference(const std::vector<float> &a, const std::vector<float> &b);

} // namespace coooda_cpp::ops
