#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/matmul.hpp>
#include <coooda_cuda/ops/matmul.cuh>

#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-3f;
constexpr float kRelTol = 1.0e-5f;

coooda_core::Tensor tensor_from_values(
    const coooda_core::Shape &shape,
    const std::vector<float> &values
) {
    if (coooda_core::numel(shape) != values.size()) {
        coooda_core::fail("matmul benchmark tensor shape does not match values");
    }

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
    const coooda_core::Tensor &expected,
    const coooda_core::Tensor &actual
) {
    if (coooda_core::to_string(expected.shape()) != coooda_core::to_string(actual.shape())) {
        coooda_core::fail(label + " shape mismatch");
    }

    const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
        label,
        tensor_values(expected),
        tensor_values(actual),
        kAbsTol,
        kRelTol
    );
    if (!report.matched) {
        coooda_core::fail(report.to_string());
    }
}

void run_matmul_compare_case() {
    const std::size_t m = 96;
    const std::size_t k = 128;
    const std::size_t n = 80;
    const coooda_core::Tensor a = tensor_from_values(
        {{m, k}},
        coooda_core::test::seeded_vector(m * k, 0xA11CEU, -2.0f, 2.0f)
    );
    const coooda_core::Tensor b = tensor_from_values(
        {{k, n}},
        coooda_core::test::seeded_vector(k * n, 0xB0BU, -2.0f, 2.0f)
    );
    const coooda_core::Tensor expected = coooda_cpp::ops::matmul_reference(a, b);

    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_matmul_baseline", [&]() {
        require_matmul_match("compare_matmul_baseline", expected, coooda_cuda::ops::matmul_baseline(a, b));
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_matmul_tiled", [&]() {
        require_matmul_match("compare_matmul_tiled", expected, coooda_cuda::ops::matmul_tiled(a, b));
    }));
}

} // namespace

namespace coooda_bench::compare {

void append_matmul_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "matmul",
        "compare C++ reference, CUDA baseline, and CUDA tiled matmul",
        []() { run_matmul_compare_case(); },
    });
}

} // namespace coooda_bench::compare
