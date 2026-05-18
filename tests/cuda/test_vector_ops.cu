#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <vector>

namespace {

void require_vector_close(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const std::string &label
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, 1.0e-6f, 1.0e-6f);
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
        {"vector_ops_cuda_add", []() {
             require_vector_close(
                 {4.0f, -1.0f, 4.5f},
                 coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f, -3.0f}, {3.0f, -3.0f, 7.5f}),
                 "vector_add_baseline"
             );
         }},

        {"vector_ops_cuda_saxpy", []() {
             require_vector_close(
                 {5.0f, -3.0f, 9.0f},
                 coooda_cuda::ops::saxpy_baseline(2.0f, {1.0f, -2.0f, 4.0f}, {3.0f, 1.0f, 1.0f}),
                 "saxpy_baseline"
             );
         }},

        {"vector_ops_cuda_relu", []() {
             require_vector_close(
                 {0.0f, 0.0f, 2.5f, 10.0f},
                 coooda_cuda::ops::relu_baseline({-3.0f, 0.0f, 2.5f, 10.0f}),
                 "relu_baseline"
             );
         }},

        {"vector_ops_cuda_sigmoid", []() {
             require_vector_close(
                 {0.11920292f, 0.5f, 0.88079708f},
                 coooda_cuda::ops::sigmoid_baseline({-2.0f, 0.0f, 2.0f}),
                 "sigmoid_baseline"
             );
         }},

        {"vector_ops_cuda_elementwise_multiply", []() {
             require_vector_close(
                 {3.0f, -8.0f, -0.0f},
                 coooda_cuda::ops::elementwise_multiply_baseline({1.0f, -2.0f, 0.0f}, {3.0f, 4.0f, -5.0f}),
                 "elementwise_multiply_baseline"
             );
         }},

        {"vector_ops_cuda_empty_inputs", []() {
             coooda_core::test::require(
                 coooda_cuda::ops::vector_add_baseline({}, {}).empty(),
                 "empty vector add"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::saxpy_baseline(3.0f, {}, {}).empty(),
                 "empty saxpy"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::relu_baseline({}).empty(),
                 "empty relu"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::sigmoid_baseline({}).empty(),
                 "empty sigmoid"
             );
             coooda_core::test::require(
                 coooda_cuda::ops::elementwise_multiply_baseline({}, {}).empty(),
                 "empty multiply"
             );
         }},

        {"vector_ops_cuda_rejects_mismatched_sizes", []() {
             require_core_error(
                 []() { (void)coooda_cuda::ops::vector_add_baseline({1.0f}, {1.0f, 2.0f}); },
                 "vector add should reject mismatched sizes"
             );
             require_core_error(
                 []() { (void)coooda_cuda::ops::saxpy_baseline(2.0f, {1.0f}, {1.0f, 2.0f}); },
                 "saxpy should reject mismatched sizes"
             );
             require_core_error(
                 []() { (void)coooda_cuda::ops::elementwise_multiply_baseline({1.0f}, {1.0f, 2.0f}); },
                 "elementwise multiply should reject mismatched sizes"
             );
         }},
    });
}
