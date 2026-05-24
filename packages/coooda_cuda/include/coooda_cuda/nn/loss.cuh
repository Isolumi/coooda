#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cuda::nn {

std::vector<float> softmax_baseline(const std::vector<float> &logits);
std::vector<float> log_softmax_baseline(const std::vector<float> &logits);

float cross_entropy_loss_baseline(
    const std::vector<float> &logits,
    std::size_t target_index
);

float mean_cross_entropy_loss_baseline(
    const std::vector<float> &logits,
    const std::vector<std::size_t> &targets,
    std::size_t class_count
);

float mean_cross_entropy_loss_device_reduce(
    const std::vector<float> &logits,
    const std::vector<std::size_t> &targets,
    std::size_t class_count
);

} // namespace coooda_cuda::nn
