#include <coooda_cpp/ops/reductions.hpp>

#include <coooda_core/status.hpp>

#include <cstddef>

namespace coooda_cpp::ops {

float sum_reference(const std::vector<float> &input) {
    float total = 0.0f;
    for (float value : input) {
        total += value;
    }
    return total;
}

float max_reference(const std::vector<float> &input) {
    if (input.empty()) {
        coooda_core::fail("max_reference requires a non-empty input");
    }

    float best = input[0];
    for (std::size_t i = 1; i < input.size(); ++i) {
        if (input[i] > best) {
            best = input[i];
        }
    }
    return best;
}

float mean_reference(const std::vector<float> &input) {
    if (input.empty()) {
        coooda_core::fail("mean_reference requires a non-empty input");
    }

    return sum_reference(input) / static_cast<float>(input.size());
}

std::size_t argmax_reference(const std::vector<float> &input) {
    if (input.empty()) {
        coooda_core::fail("argmax_reference requires a non-empty input");
    }

    std::size_t best_index = 0;
    for (std::size_t i = 1; i < input.size(); ++i) {
        if (input[i] > input[best_index]) {
            best_index = i;
        }
    }
    return best_index;
}

} // namespace coooda_cpp::ops
