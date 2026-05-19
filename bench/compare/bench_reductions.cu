#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/reductions.hpp>
#include <coooda_cuda/ops/reductions.cuh>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-2f;
constexpr float kRelTol = 1.0e-5f;

void require_close(const std::string &label, float expected, float actual) {
    if (coooda_core::test::close_enough(expected, actual, kAbsTol, kRelTol)) {
        return;
    }

    std::ostringstream message;
    message << label << " expected " << expected << ", actual " << actual;
    coooda_core::fail(message.str());
}

void require_same_argmax(const std::vector<float> &input) {
    const std::size_t expected = coooda_cpp::ops::argmax_reference(input);
    const std::size_t actual = coooda_cuda::ops::argmax_baseline(input);
    if (expected == actual) {
        return;
    }

    std::ostringstream message;
    message << "argmax expected " << expected << ", actual " << actual;
    coooda_core::fail(message.str());
}

void compare_reductions(const std::vector<float> &input) {
    require_close(
        "sum",
        coooda_cpp::ops::sum_reference(input),
        coooda_cuda::ops::sum_baseline(input)
    );
    require_close(
        "max",
        coooda_cpp::ops::max_reference(input),
        coooda_cuda::ops::max_baseline(input)
    );
    require_close(
        "mean",
        coooda_cpp::ops::mean_reference(input),
        coooda_cuda::ops::mean_baseline(input)
    );
    require_same_argmax(input);
}

void run_reductions_compare_case() {
    const std::vector<float> input =
        coooda_core::test::seeded_vector(16 * 1024, 0x5A5AU, -10.0f, 10.0f);

    const auto result = coooda_core::bench::time_once("compare_reductions", [&]() {
        compare_reductions(input);
    });
    coooda_core::bench::print_result(result);
}

} // namespace

namespace coooda_bench::compare {

void append_reductions_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "reductions",
        "compare C++ reference and CUDA baseline reductions",
        []() { run_reductions_compare_case(); },
    });
}

} // namespace coooda_bench::compare
