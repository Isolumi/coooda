#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cpp::ops {

float sum_reference(const std::vector<float> &input);
float max_reference(const std::vector<float> &input);
float mean_reference(const std::vector<float> &input);
std::size_t argmax_reference(const std::vector<float> &input);

} // namespace coooda_cpp::ops
