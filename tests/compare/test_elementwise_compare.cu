#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/elementwise.hpp>
#include <coooda_cuda/ops/elementwise.cuh>

#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-4f;
constexpr float kRelTol = 1.0e-5f;

void require_vector_close(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

void compare_unary_case(const std::string &label, const std::vector<float> &input) {
    const std::vector<float> relu_expected = coooda_cpp::ops::unary_relu_reference(input);
    require_vector_close(
        label + "_relu_baseline",
        relu_expected,
        coooda_cuda::ops::unary_relu_baseline(input)
    );
    require_vector_close(
        label + "_relu_grid_stride",
        relu_expected,
        coooda_cuda::ops::unary_relu_grid_stride(input)
    );

    const std::vector<float> gelu_expected = coooda_cpp::ops::unary_gelu_reference(input);
    require_vector_close(
        label + "_gelu_baseline",
        gelu_expected,
        coooda_cuda::ops::unary_gelu_baseline(input)
    );
    require_vector_close(
        label + "_gelu_grid_stride",
        gelu_expected,
        coooda_cuda::ops::unary_gelu_grid_stride(input)
    );
}

void compare_binary_case(
    const std::string &label,
    const std::vector<float> &a,
    const std::vector<float> &b
) {
    const std::vector<float> add_expected = coooda_cpp::ops::binary_add_reference(a, b);
    require_vector_close(
        label + "_add_baseline",
        add_expected,
        coooda_cuda::ops::binary_add_baseline(a, b)
    );
    require_vector_close(
        label + "_add_grid_stride",
        add_expected,
        coooda_cuda::ops::binary_add_grid_stride(a, b)
    );

    const std::vector<float> multiply_expected = coooda_cpp::ops::binary_multiply_reference(a, b);
    require_vector_close(
        label + "_multiply_baseline",
        multiply_expected,
        coooda_cuda::ops::binary_multiply_baseline(a, b)
    );
    require_vector_close(
        label + "_multiply_grid_stride",
        multiply_expected,
        coooda_cuda::ops::binary_multiply_grid_stride(a, b)
    );
}

void compare_broadcast_case(
    const std::string &label,
    const std::vector<float> &input,
    const std::vector<float> &bias
) {
    const std::vector<float> scalar_expected = coooda_cpp::ops::add_scalar_reference(input, 1.25f);
    require_vector_close(
        label + "_scalar_baseline",
        scalar_expected,
        coooda_cuda::ops::add_scalar_baseline(input, 1.25f)
    );
    require_vector_close(
        label + "_scalar_grid_stride",
        scalar_expected,
        coooda_cuda::ops::add_scalar_grid_stride(input, 1.25f)
    );

    const std::vector<float> bias_expected = coooda_cpp::ops::add_bias_reference(input, bias);
    require_vector_close(
        label + "_bias_baseline",
        bias_expected,
        coooda_cuda::ops::add_bias_baseline(input, bias)
    );
    require_vector_close(
        label + "_bias_grid_stride",
        bias_expected,
        coooda_cuda::ops::add_bias_grid_stride(input, bias)
    );

    const std::vector<float> relu_expected = coooda_cpp::ops::add_bias_relu_reference(input, bias);
    require_vector_close(
        label + "_bias_relu_baseline",
        relu_expected,
        coooda_cuda::ops::add_bias_relu_baseline(input, bias)
    );
    require_vector_close(
        label + "_bias_relu_grid_stride",
        relu_expected,
        coooda_cuda::ops::add_bias_relu_grid_stride(input, bias)
    );

    const std::vector<float> gelu_expected = coooda_cpp::ops::add_bias_gelu_reference(input, bias);
    require_vector_close(
        label + "_bias_gelu_baseline",
        gelu_expected,
        coooda_cuda::ops::add_bias_gelu_baseline(input, bias)
    );
    require_vector_close(
        label + "_bias_gelu_grid_stride",
        gelu_expected,
        coooda_cuda::ops::add_bias_gelu_grid_stride(input, bias)
    );
}

std::vector<float> seeded(std::size_t size, std::uint32_t seed) {
    return coooda_core::test::seeded_vector(size, seed, -3.0f, 3.0f);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"elementwise_compare_empty_cases", []() {
             compare_unary_case("empty_unary", {});
             compare_binary_case("empty_binary", {}, {});
             compare_broadcast_case("empty_broadcast", {}, {1.0f, 2.0f});
         }},

        {"elementwise_compare_small_cases", []() {
             compare_unary_case("small_unary", {-3.0f, 0.0f, 2.5f, 10.0f});
             compare_binary_case(
                 "small_binary",
                 {1.0f, 2.0f, -3.0f},
                 {3.0f, -3.0f, 7.5f}
             );
             compare_broadcast_case(
                 "small_broadcast",
                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                 {0.5f, -1.0f, 2.0f}
             );
         }},

        {"elementwise_compare_seeded_cases", []() {
             compare_unary_case("seeded_unary", seeded(1024, 0xA11CEU));
             compare_binary_case("seeded_binary", seeded(1024, 0xB0BU), seeded(1024, 0xC001U));
             compare_broadcast_case("seeded_broadcast", seeded(1536, 0xD00DU), seeded(48, 0x5A5AU));
         }},

        {"elementwise_compare_non_grid_multiple", []() {
             compare_unary_case("non_grid_unary", seeded(513, 0x1234U));
             compare_binary_case("non_grid_binary", seeded(513, 0x2345U), seeded(513, 0x3456U));
             compare_broadcast_case("non_grid_broadcast", seeded(765, 0x4567U), seeded(15, 0x5678U));
         }},
    });
}
