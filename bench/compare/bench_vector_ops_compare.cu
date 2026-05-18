#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>

#include <vector>

namespace {

void run_vector_ops_compare_case() {
    const auto result = coooda_core::bench::time_once("compare_vector_ops", []() {
        if (!coooda_compare::ops::vector_ops_match_reference()) {
            coooda_core::fail("vector_ops compare benchmark found a backend mismatch");
        }
    });
    coooda_core::bench::print_result(result);
}

} // namespace

namespace coooda_bench::compare {

void append_vector_ops_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "vector_ops",
        "compare C++ reference, CUDA baseline, and CUDA grid-stride vector ops",
        []() { run_vector_ops_compare_case(); },
    });
}

} // namespace coooda_bench::compare
