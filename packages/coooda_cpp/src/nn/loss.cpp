#include <coooda_cpp/nn/loss.hpp>

#include <coooda_core/status.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

void require_non_empty_logits(std::size_t size, const char *operation) {
    if (size == 0) {
        coooda_core::fail(std::string(operation) + " requires non-empty logits");
    }
}

void require_target_in_range(std::size_t target_index, std::size_t class_count, const char *operation) {
    if (target_index >= class_count) {
        coooda_core::fail(std::string(operation) + " target index is out of range");
    }
}

void require_batch_shape(
    std::size_t logits_size,
    std::size_t target_count,
    std::size_t class_count,
    const char *operation
) {
    if (target_count == 0) {
        coooda_core::fail(std::string(operation) + " requires at least one target");
    }
    if (class_count == 0) {
        coooda_core::fail(std::string(operation) + " requires non-zero class count");
    }
    if (target_count > (std::numeric_limits<std::size_t>::max() / class_count)) {
        coooda_core::fail(std::string(operation) + " batch shape is too large");
    }
    if (logits_size != target_count * class_count) {
        coooda_core::fail(std::string(operation) + " logits size must equal target count times class count");
    }
}

float max_value(const std::vector<float> &logits, std::size_t offset, std::size_t count) {
    float max_logit = logits[offset];
    for (std::size_t i = 1; i < count; ++i) {
        max_logit = std::max(max_logit, logits[offset + i]);
    }
    return max_logit;
}

float log_sum_exp_shifted(
    const std::vector<float> &logits,
    std::size_t offset,
    std::size_t count,
    float max_logit
) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        sum += std::exp(logits[offset + i] - max_logit);
    }
    return std::log(sum);
}

float cross_entropy_row(
    const std::vector<float> &logits,
    std::size_t offset,
    std::size_t class_count,
    std::size_t target_index
) {
    const float max_logit = max_value(logits, offset, class_count);
    const float shifted_log_sum_exp = log_sum_exp_shifted(logits, offset, class_count, max_logit);
    return (max_logit + shifted_log_sum_exp) - logits[offset + target_index];
}

} // namespace

namespace coooda_cpp::nn {

std::vector<float> softmax_reference(const std::vector<float> &logits) {
    require_non_empty_logits(logits.size(), "softmax_reference");

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    std::vector<float> out(logits.size(), 0.0f);

    for (std::size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(logits[i] - max_logit);
        sum += out[i];
    }
    for (float &value : out) {
        value /= sum;
    }
    return out;
}

std::vector<float> log_softmax_reference(const std::vector<float> &logits) {
    require_non_empty_logits(logits.size(), "log_softmax_reference");

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    const float shifted_log_sum_exp = log_sum_exp_shifted(logits, 0, logits.size(), max_logit);

    std::vector<float> out(logits.size(), 0.0f);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        out[i] = logits[i] - max_logit - shifted_log_sum_exp;
    }
    return out;
}

float cross_entropy_loss_reference(
    const std::vector<float> &logits,
    std::size_t target_index
) {
    require_non_empty_logits(logits.size(), "cross_entropy_loss_reference");
    require_target_in_range(target_index, logits.size(), "cross_entropy_loss_reference");

    return cross_entropy_row(logits, 0, logits.size(), target_index);
}

float mean_cross_entropy_loss_reference(
    const std::vector<float> &logits,
    const std::vector<std::size_t> &targets,
    std::size_t class_count
) {
    require_batch_shape(logits.size(), targets.size(), class_count, "mean_cross_entropy_loss_reference");

    float total = 0.0f;
    for (std::size_t row = 0; row < targets.size(); ++row) {
        require_target_in_range(targets[row], class_count, "mean_cross_entropy_loss_reference");
        total += cross_entropy_row(logits, row * class_count, class_count, targets[row]);
    }
    return total / static_cast<float>(targets.size());
}

} // namespace coooda_cpp::nn
