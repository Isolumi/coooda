#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/normalization.hpp>
#include <coooda_cuda/nn/normalization.cuh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-4f;
constexpr float kRelTol = 1.0e-5f;
constexpr float kEpsilon = 1.0e-5f;

void require_vector_close(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

void compare_case(
    const std::string &label,
    const std::vector<float> &input,
    const std::vector<float> &gamma,
    const std::vector<float> &beta,
    std::size_t feature_count
) {
    const std::vector<float> layer_expected =
        coooda_cpp::nn::layer_norm_reference(input, gamma, beta, feature_count, kEpsilon);
    require_vector_close(
        label + "_layer_norm_baseline",
        layer_expected,
        coooda_cuda::nn::layer_norm_baseline(input, gamma, beta, feature_count, kEpsilon)
    );
    require_vector_close(
        label + "_layer_norm_welford",
        layer_expected,
        coooda_cuda::nn::layer_norm_welford(input, gamma, beta, feature_count, kEpsilon)
    );

    require_vector_close(
        label + "_rms_norm_baseline",
        coooda_cpp::nn::rms_norm_reference(input, gamma, feature_count, kEpsilon),
        coooda_cuda::nn::rms_norm_baseline(input, gamma, feature_count, kEpsilon)
    );
}

std::vector<float> seeded(std::size_t size, std::uint32_t seed, float low = -4.0f, float high = 4.0f) {
    return coooda_core::test::seeded_vector(size, seed, low, high);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"normalization_compare_known_values", []() {
             compare_case(
                 "known",
                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                 {2.0f, 1.0f, 0.5f},
                 {0.0f, 0.5f, 0.0f},
                 3
             );
         }},

        {"normalization_compare_single_row", []() {
             compare_case(
                 "single_row",
                 seeded(17, 0x901U),
                 seeded(17, 0x902U, 0.5f, 1.5f),
                 seeded(17, 0x903U, -0.25f, 0.25f),
                 17
             );
         }},

        {"normalization_compare_seeded_rows", []() {
             compare_case(
                 "seeded_rows",
                 seeded(16 * 64, 0x904U),
                 seeded(64, 0x905U, 0.5f, 1.5f),
                 seeded(64, 0x906U, -0.25f, 0.25f),
                 64
             );
         }},

        {"normalization_compare_wide_rows", []() {
             compare_case(
                 "wide_rows",
                 seeded(4 * 513, 0x907U),
                 seeded(513, 0x908U, 0.5f, 1.5f),
                 seeded(513, 0x909U, -0.25f, 0.25f),
                 513
             );
         }},
    });
}
