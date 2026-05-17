#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>

#include <vector>

namespace {

void run_core_helpers_case() {
    const coooda_core::Shape shape{{64, 64}};
    if (coooda_core::numel(shape) != 4096) {
        coooda_core::fail("core_helpers benchmark shape numel check failed");
    }

    const std::vector<float> expected =
        coooda_core::test::seeded_vector(coooda_core::numel(shape), 0xC0DAU, -1.0f, 1.0f);
    std::vector<float> actual = expected;
    actual[actual.size() / 2] += 0.25f;

    const auto result = coooda_core::bench::time_once("core_helpers_compare_vectors", [&]() {
        const coooda_core::test::MismatchReport matched = coooda_core::test::compare_vectors(
            "core_helpers_match",
            expected,
            expected,
            1.0e-6f,
            1.0e-6f
        );
        if (!matched.matched) {
            coooda_core::fail(matched.to_string());
        }

        const coooda_core::test::MismatchReport mismatched = coooda_core::test::compare_vectors(
            "core_helpers_mismatch",
            expected,
            actual,
            1.0e-6f,
            1.0e-6f
        );
        if (mismatched.matched) {
            coooda_core::fail("core_helpers benchmark expected a mismatch report");
        }
    });

    coooda_core::bench::print_result(result);
}

} // namespace

namespace coooda_bench::compare {

void append_core_helper_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "core_helpers",
        "core helper shape and mismatch-report benchmark",
        []() { run_core_helpers_case(); },
    });
}

} // namespace coooda_bench::compare
