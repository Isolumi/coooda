#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/matmul.hpp>
#include <coooda_cuda/ops/matmul.cuh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

coooda_core::Tensor tensor_from_values(
    const coooda_core::Shape &shape,
    const std::vector<float> &values
) {
    coooda_core::test::require(
        coooda_core::numel(shape) == values.size(),
        "tensor_from_values shape must match values"
    );

    coooda_core::Tensor tensor(shape);
    for (std::size_t i = 0; i < values.size(); ++i) {
        tensor[i] = values[i];
    }
    return tensor;
}

std::vector<float> tensor_values(const coooda_core::Tensor &tensor) {
    std::vector<float> values(tensor.size(), 0.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = tensor[i];
    }
    return values;
}

void require_matmul_match(
    const std::string &label,
    const coooda_core::Tensor &a,
    const coooda_core::Tensor &b
) {
    const coooda_core::Tensor expected = coooda_cpp::ops::matmul_reference(a, b);
    const std::vector<std::pair<std::string, coooda_core::Tensor>> actuals{
        {"baseline", coooda_cuda::ops::matmul_baseline(a, b)},
        {"tiled", coooda_cuda::ops::matmul_tiled(a, b)},
    };

    for (const auto &[variant, actual] : actuals) {
        coooda_core::test::require_equal(
            coooda_core::to_string(expected.shape()),
            coooda_core::to_string(actual.shape()),
            label + " " + variant + " shape"
        );

        const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
            label + " " + variant,
            tensor_values(expected),
            tensor_values(actual),
            1.0e-4f,
            1.0e-5f
        );
        coooda_core::test::require(report.matched, report.to_string());
    }
}

void require_seeded_matmul_match(
    const std::string &label,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::uint32_t seed_a,
    std::uint32_t seed_b
) {
    const coooda_core::Tensor a = tensor_from_values(
        {{m, k}},
        coooda_core::test::seeded_vector(m * k, seed_a, -2.0f, 2.0f)
    );
    const coooda_core::Tensor b = tensor_from_values(
        {{k, n}},
        coooda_core::test::seeded_vector(k * n, seed_b, -2.0f, 2.0f)
    );

    require_matmul_match(label, a, b);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"matmul_compare_known_values", []() {
             const coooda_core::Tensor a = tensor_from_values(
                 {{2, 3}},
                 {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f}
             );
             const coooda_core::Tensor b = tensor_from_values(
                 {{3, 2}},
                 {7.0f, 8.0f,
                  9.0f, 10.0f,
                  11.0f, 12.0f}
             );

             require_matmul_match("known_values", a, b);
         }},

        {"matmul_compare_rectangular", []() {
             const coooda_core::Tensor a = tensor_from_values(
                 {{3, 2}},
                 {1.0f, 2.0f,
                  -1.0f, 0.5f,
                  3.0f, -4.0f}
             );
             const coooda_core::Tensor b = tensor_from_values(
                 {{2, 4}},
                 {2.0f, 0.0f, -1.0f, 3.0f,
                  1.5f, 4.0f, 2.0f, -2.0f}
             );

             require_matmul_match("rectangular", a, b);
         }},

        {"matmul_compare_empty_output", []() {
             require_matmul_match(
                 "empty_output",
                 coooda_core::Tensor({{0, 3}}),
                 coooda_core::Tensor({{3, 2}})
             );
         }},

        {"matmul_compare_zero_inner_dimension", []() {
             require_matmul_match(
                 "zero_inner_dimension",
                 coooda_core::Tensor({{2, 0}}),
                 coooda_core::Tensor({{0, 4}})
             );
         }},

        {"matmul_compare_seeded_small", []() {
             require_seeded_matmul_match("seeded_small", 4, 5, 3, 0xA11CEU, 0xB0BU);
         }},

        {"matmul_compare_seeded_medium", []() {
             require_seeded_matmul_match("seeded_medium", 16, 17, 9, 0xC001U, 0xD00DU);
         }},
    });
}
