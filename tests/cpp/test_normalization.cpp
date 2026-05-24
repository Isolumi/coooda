#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/normalization.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-5f;
constexpr float kRelTol = 1.0e-5f;
constexpr float kEpsilon = 1.0e-5f;

void require_vector_close(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const std::string &label
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

template <typename Fn>
void require_core_error(Fn &&fn, const std::string &message) {
    bool threw = false;
    try {
        fn();
    } catch (const coooda_core::Error &) {
        threw = true;
    }
    coooda_core::test::require(threw, message);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"normalization_cpp_layer_norm_known_values", []() {
             require_vector_close(
                 {-1.224735f, 0.0f, 1.224735f},
                 coooda_cpp::nn::layer_norm_reference(
                     {1.0f, 2.0f, 3.0f},
                     {1.0f, 1.0f, 1.0f},
                     {0.0f, 0.0f, 0.0f},
                     3,
                     kEpsilon
                 ),
                 "layer norm known values"
             );
         }},

        {"normalization_cpp_layer_norm_affine_and_batch", []() {
             require_vector_close(
                 {-2.449471f, 0.5f, 0.612368f, -2.449471f, 0.5f, 0.612368f},
                 coooda_cpp::nn::layer_norm_reference(
                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                     {2.0f, 1.0f, 0.5f},
                     {0.0f, 0.5f, 0.0f},
                     3,
                     kEpsilon
                 ),
                 "layer norm affine batch"
             );
         }},

        {"normalization_cpp_rms_norm_known_values", []() {
             require_vector_close(
                 {0.46291f, 0.92582f, 1.38873f},
                 coooda_cpp::nn::rms_norm_reference(
                     {1.0f, 2.0f, 3.0f},
                     {1.0f, 1.0f, 1.0f},
                     3,
                     kEpsilon
                 ),
                 "rms norm known values"
             );
         }},

        {"normalization_cpp_rms_norm_affine_and_batch", []() {
             require_vector_close(
                 {0.925819f, 0.925819f, 0.694364f, 1.579084f, 0.986928f, 0.592157f},
                 coooda_cpp::nn::rms_norm_reference(
                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                     {2.0f, 1.0f, 0.5f},
                     3,
                     kEpsilon
                 ),
                 "rms norm affine batch"
             );
         }},

        {"normalization_cpp_empty_inputs", []() {
             coooda_core::test::require(
                 coooda_cpp::nn::layer_norm_reference({}, {1.0f, 1.0f}, {0.0f, 0.0f}, 2).empty(),
                 "empty layer norm"
             );
             coooda_core::test::require(
                 coooda_cpp::nn::rms_norm_reference({}, {1.0f, 1.0f}, 2).empty(),
                 "empty rms norm"
             );
         }},

        {"normalization_cpp_seeded_inputs_are_finite", []() {
             const std::vector<float> input =
                 coooda_core::test::seeded_vector(24, 0x901U, -4.0f, 4.0f);
             const std::vector<float> gamma =
                 coooda_core::test::seeded_vector(6, 0x902U, 0.5f, 1.5f);
             const std::vector<float> beta =
                 coooda_core::test::seeded_vector(6, 0x903U, -0.25f, 0.25f);
             const std::vector<float> layer =
                 coooda_cpp::nn::layer_norm_reference(input, gamma, beta, 6);
             const std::vector<float> rms =
                 coooda_cpp::nn::rms_norm_reference(input, gamma, 6);

             for (const float value : layer) {
                 coooda_core::test::require(std::isfinite(value), "layer norm should stay finite");
             }
             for (const float value : rms) {
                 coooda_core::test::require(std::isfinite(value), "rms norm should stay finite");
             }
         }},

        {"normalization_cpp_rejects_invalid_shapes", []() {
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::layer_norm_reference(
                         {1.0f, 2.0f, 3.0f},
                         {1.0f, 1.0f},
                         {0.0f, 0.0f},
                         2
                     );
                 },
                 "layer norm should reject non-divisible input size"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::layer_norm_reference(
                         {1.0f, 2.0f},
                         {1.0f},
                         {0.0f, 0.0f},
                         2
                     );
                 },
                 "layer norm should reject wrong gamma size"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::rms_norm_reference(
                         {1.0f, 2.0f},
                         {1.0f, 1.0f},
                         0
                     );
                 },
                 "rms norm should reject zero feature count"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::rms_norm_reference(
                         {1.0f, 2.0f},
                         {1.0f, 1.0f},
                         2,
                         0.0f
                     );
                 },
                 "rms norm should reject non-positive epsilon"
             );
         }},
    });
}
