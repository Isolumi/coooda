#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cpp::nn {

std::vector<float> layer_norm_reference(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    const std::vector<float> &beta,
    std::size_t feature_count,
    float epsilon = 1.0e-5f
);

std::vector<float> rms_norm_reference(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    std::size_t feature_count,
    float epsilon = 1.0e-5f
);

} // namespace coooda_cpp::nn
