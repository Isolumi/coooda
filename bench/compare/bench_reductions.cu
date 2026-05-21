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

struct ReductionExpected {
    float sum = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    std::size_t argmax = 0;
};

ReductionExpected expected_reductions(const std::vector<float> &input) {
    return ReductionExpected{
        coooda_cpp::ops::sum_reference(input),
        coooda_cpp::ops::max_reference(input),
        coooda_cpp::ops::mean_reference(input),
        coooda_cpp::ops::argmax_reference(input),
    };
}

void require_same_argmax(std::size_t expected, std::size_t actual) {
    if (expected == actual) {
        return;
    }

    std::ostringstream message;
    message << "argmax expected " << expected << ", actual " << actual;
    coooda_core::fail(message.str());
}

void compare_reductions_baseline(const ReductionExpected &expected, const std::vector<float> &input) {
    require_close("baseline sum", expected.sum, coooda_cuda::ops::sum_baseline(input));
    require_close("baseline max", expected.max, coooda_cuda::ops::max_baseline(input));
    require_close("baseline mean", expected.mean, coooda_cuda::ops::mean_baseline(input));
    require_same_argmax(expected.argmax, coooda_cuda::ops::argmax_baseline(input));
}

void compare_reductions_device_reduce(const ReductionExpected &expected, const std::vector<float> &input) {
    require_close("device_reduce sum", expected.sum, coooda_cuda::ops::sum_device_reduce(input));
    require_close("device_reduce max", expected.max, coooda_cuda::ops::max_device_reduce(input));
    require_close("device_reduce mean", expected.mean, coooda_cuda::ops::mean_device_reduce(input));
    require_same_argmax(expected.argmax, coooda_cuda::ops::argmax_device_reduce(input));
}

void run_reductions_compare_case() {
    const std::vector<float> input =
        coooda_core::test::seeded_vector(16 * 1024, 0x5A5AU, -10.0f, 10.0f);
    const ReductionExpected expected = expected_reductions(input);

    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_reductions_baseline", [&]() {
        compare_reductions_baseline(expected, input);
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_reductions_device_reduce", [&]() {
        compare_reductions_device_reduce(expected, input);
    }));
}

} // namespace

namespace coooda_bench::compare {

void append_reductions_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "reductions",
        "compare C++ reference, CUDA baseline, and CUDA device-final reductions",
        []() { run_reductions_compare_case(); },
    });
}

} // namespace coooda_bench::compare
