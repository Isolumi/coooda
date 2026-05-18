#pragma once

#include <vector>

namespace coooda_cuda::ops {

std::vector<float> vector_add_baseline(const std::vector<float> &a, const std::vector<float> &b);
std::vector<float> saxpy_baseline(float alpha, const std::vector<float> &x, const std::vector<float> &y);
std::vector<float> relu_baseline(const std::vector<float> &input);
std::vector<float> sigmoid_baseline(const std::vector<float> &input);
std::vector<float> elementwise_multiply_baseline(const std::vector<float> &a, const std::vector<float> &b);

std::vector<float> vector_add_grid_stride(const std::vector<float> &a, const std::vector<float> &b);
std::vector<float> saxpy_grid_stride(float alpha, const std::vector<float> &x, const std::vector<float> &y);
std::vector<float> relu_grid_stride(const std::vector<float> &input);
std::vector<float> sigmoid_grid_stride(const std::vector<float> &input);
std::vector<float> elementwise_multiply_grid_stride(const std::vector<float> &a, const std::vector<float> &b);

} // namespace coooda_cuda::ops
