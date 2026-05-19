#include <coooda_core/status.hpp>
#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/matmul.hpp>

#include <cstddef>
#include <string>
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

void require_tensor_close(
    const coooda_core::Tensor &actual,
    const coooda_core::Shape &expected_shape,
    const std::vector<float> &expected_values,
    const std::string &label
) {
    coooda_core::test::require_equal(
        coooda_core::to_string(expected_shape),
        coooda_core::to_string(actual.shape()),
        label + " shape"
    );

    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected_values, tensor_values(actual), 1.0e-6f, 1.0e-6f);
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
        {"matmul_cpp_known_values", []() {
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

             require_tensor_close(
                 coooda_cpp::ops::matmul_reference(a, b),
                 {{2, 2}},
                 {58.0f, 64.0f,
                  139.0f, 154.0f},
                 "known matmul"
             );
         }},

        {"matmul_cpp_rectangular_output", []() {
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

             require_tensor_close(
                 coooda_cpp::ops::matmul_reference(a, b),
                 {{3, 4}},
                 {5.0f, 8.0f, 3.0f, -1.0f,
                  -1.25f, 2.0f, 2.0f, -4.0f,
                  0.0f, -16.0f, -11.0f, 17.0f},
                 "rectangular matmul"
             );
         }},

        {"matmul_cpp_allows_empty_2d_output", []() {
             const coooda_core::Tensor a({{0, 3}});
             const coooda_core::Tensor b({{3, 2}});

             require_tensor_close(
                 coooda_cpp::ops::matmul_reference(a, b),
                 {{0, 2}},
                 {},
                 "empty matmul"
             );
         }},

        {"matmul_cpp_rejects_non_matrix_inputs", []() {
             require_core_error(
                 []() {
                     const coooda_core::Tensor a({{2, 3, 1}});
                     const coooda_core::Tensor b({{3, 2}});
                     (void)coooda_cpp::ops::matmul_reference(a, b);
                 },
                 "left input must be rank 2"
             );
             require_core_error(
                 []() {
                     const coooda_core::Tensor a({{2, 3}});
                     const coooda_core::Tensor b({{3}});
                     (void)coooda_cpp::ops::matmul_reference(a, b);
                 },
                 "right input must be rank 2"
             );
         }},

        {"matmul_cpp_rejects_inner_dimension_mismatch", []() {
             require_core_error(
                 []() {
                     const coooda_core::Tensor a({{2, 3}});
                     const coooda_core::Tensor b({{4, 2}});
                     (void)coooda_cpp::ops::matmul_reference(a, b);
                 },
                 "inner dimensions must match"
             );
         }},
    });
}
