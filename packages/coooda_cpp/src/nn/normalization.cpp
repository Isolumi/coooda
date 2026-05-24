#include <coooda_cpp/nn/normalization.hpp>

#include <coooda_core/status.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

void require_positive_epsilon(float epsilon, const char *operation) {
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        coooda_core::fail(std::string(operation) + " requires positive finite epsilon");
    }
}

void require_norm_shape(
    std::size_t input_size,
    std::size_t gamma_size,
    std::size_t feature_count,
    float epsilon,
    const char *operation
) {
    if (feature_count == 0) {
        coooda_core::fail(std::string(operation) + " requires non-zero feature count");
    }
    if (gamma_size != feature_count) {
        coooda_core::fail(std::string(operation) + " gamma size must equal feature count");
    }
    if (input_size % feature_count != 0) {
        coooda_core::fail(std::string(operation) + " input size must be divisible by feature count");
    }
    require_positive_epsilon(epsilon, operation);
}

void require_layer_norm_shape(
    std::size_t input_size,
    std::size_t gamma_size,
    std::size_t beta_size,
    std::size_t feature_count,
    float epsilon,
    const char *operation
) {
    require_norm_shape(input_size, gamma_size, feature_count, epsilon, operation);
    if (beta_size != feature_count) {
        coooda_core::fail(std::string(operation) + " beta size must equal feature count");
    }
}

} // namespace

namespace coooda_cpp::nn {

std::vector<float> layer_norm_reference(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    const std::vector<float> &beta,
    std::size_t feature_count,
    float epsilon
) {
    require_layer_norm_shape(
        input.size(),
        gamma.size(),
        beta.size(),
        feature_count,
        epsilon,
        "layer_norm_reference"
    );

    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t row = 0; row < input.size(); row += feature_count) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < feature_count; ++col) {
            sum += input[row + col];
        }
        const float mean = sum / static_cast<float>(feature_count);

        float variance_sum = 0.0f;
        for (std::size_t col = 0; col < feature_count; ++col) {
            const float centered = input[row + col] - mean;
            variance_sum += centered * centered;
        }
        const float variance = variance_sum / static_cast<float>(feature_count);
        const float inv_std = 1.0f / std::sqrt(variance + epsilon);

        for (std::size_t col = 0; col < feature_count; ++col) {
            out[row + col] = ((input[row + col] - mean) * inv_std * gamma[col]) + beta[col];
        }
    }
    return out;
}

std::vector<float> rms_norm_reference(
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    std::size_t feature_count,
    float epsilon
) {
    require_norm_shape(input.size(), gamma.size(), feature_count, epsilon, "rms_norm_reference");

    std::vector<float> out(input.size(), 0.0f);
    for (std::size_t row = 0; row < input.size(); row += feature_count) {
        float square_sum = 0.0f;
        for (std::size_t col = 0; col < feature_count; ++col) {
            square_sum += input[row + col] * input[row + col];
        }

        const float mean_square = square_sum / static_cast<float>(feature_count);
        const float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t col = 0; col < feature_count; ++col) {
            out[row + col] = input[row + col] * inv_rms * gamma[col];
        }
    }
    return out;
}

} // namespace coooda_cpp::nn
