#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cuda::ops {

float sum_baseline(const std::vector<float> &input);
float max_baseline(const std::vector<float> &input);
float mean_baseline(const std::vector<float> &input);
std::size_t argmax_baseline(const std::vector<float> &input);

} // namespace coooda_cuda::ops
