#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/elementwise.cuh>

#include <string>
#include <vector>

namespace {

void require_vector_close(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const std::string &label
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, 1.0e-5f, 1.0e-5f);
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
        {"elementwise_cuda_unary_relu", []() {
             require_vector_close(
                 {0.0f, 0.0f, 2.5f, 10.0f},
                 coooda_cuda::ops::unary_relu_baseline({-3.0f, 0.0f, 2.5f, 10.0f}),
                 "unary relu"
             );
         }},

        {"elementwise_cuda_unary_gelu", []() {
             require_vector_close(
                 {-0.158808f, 0.0f, 0.841192f, 1.954598f},
                 coooda_cuda::ops::unary_gelu_baseline({-1.0f, 0.0f, 1.0f, 2.0f}),
                 "unary gelu"
             );
         }},

        {"elementwise_cuda_binary_ops", []() {
             require_vector_close(
                 {4.0f, -1.0f, 4.5f},
                 coooda_cuda::ops::binary_add_baseline({1.0f, 2.0f, -3.0f}, {3.0f, -3.0f, 7.5f}),
                 "binary add"
             );
             require_vector_close(
                 {3.0f, -8.0f, -0.0f},
                 coooda_cuda::ops::binary_multiply_baseline({1.0f, -2.0f, 0.0f}, {3.0f, 4.0f, -5.0f}),
                 "binary multiply"
             );
         }},

        {"elementwise_cuda_broadcast_like_ops", []() {
             require_vector_close(
                 {0.5f, 3.5f, 5.0f},
                 coooda_cuda::ops::add_scalar_baseline({-1.0f, 2.0f, 3.5f}, 1.5f),
                 "add scalar"
             );
             require_vector_close(
                 {1.5f, 1.0f, 5.0f, 4.5f, 4.0f, 8.0f},
                 coooda_cuda::ops::add_bias_baseline(
                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                     {0.5f, -1.0f, 2.0f}
                 ),
                 "add bias"
             );
         }},

        {"elementwise_cuda_fused_bias_activation", []() {
             require_vector_close(
                 {1.5f, 0.0f, 5.0f, 4.5f, 2.0f, 8.0f},
                 coooda_cuda::ops::add_bias_relu_baseline(
                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                     {0.5f, -3.0f, 2.0f}
                 ),
                 "add bias relu"
             );
             require_vector_close(
                 {-0.158808f, 0.0f, 0.841192f, 1.954598f},
                 coooda_cuda::ops::add_bias_gelu_baseline(
                     {-1.0f, 0.0f, 1.0f, 2.0f},
                     {0.0f, 0.0f}
                 ),
                 "add bias gelu"
             );
         }},

        {"elementwise_cuda_empty_inputs", []() {
             coooda_core::test::require(
                 coooda_cuda::ops::unary_relu_baseline({}).empty(),
                 "empty relu"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::unary_gelu_baseline({}).empty(),
                 "empty gelu"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::binary_add_baseline({}, {}).empty(),
                 "empty binary add"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::add_scalar_baseline({}, 1.0f).empty(),
                 "empty scalar add"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::add_bias_baseline({}, {1.0f, 2.0f}).empty(),
                 "empty bias add"
             );
         }},

        {"elementwise_cuda_rejects_invalid_shapes", []() {
             require_core_error(
                 []() { (void)coooda_cuda::ops::binary_add_baseline({1.0f}, {1.0f, 2.0f}); },
                 "binary add should reject mismatched sizes"
             );
             require_core_error(
                 []() { (void)coooda_cuda::ops::binary_multiply_baseline({1.0f}, {1.0f, 2.0f}); },
                 "binary multiply should reject mismatched sizes"
             );
             require_core_error(
                 []() { (void)coooda_cuda::ops::add_bias_baseline({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f}); },
                 "bias add should reject non-divisible input size"
             );
             require_core_error(
                 []() { (void)coooda_cuda::ops::add_bias_relu_baseline({1.0f}, {}); },
                 "bias activation should reject empty bias with non-empty input"
             );
         }},
    });
}
