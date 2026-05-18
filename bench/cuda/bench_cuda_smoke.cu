#include <coooda_core/benchmark.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <vector>

namespace coooda_bench::cuda {

void append_vector_ops_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);

} // namespace coooda_bench::cuda

int main(int argc, char **argv) {
    std::vector<coooda_core::bench::BenchmarkCase> cases{
        {"smoke",
         "tiny CUDA vector-add smoke benchmark",
         []() {
             const auto result = coooda_core::bench::time_once("cuda_vector_add_baseline_smoke", []() {
                 (void)coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f}, {3.0f, 4.0f});
             });
             coooda_core::bench::print_result(result);
         }},
    };

    coooda_bench::cuda::append_vector_ops_cases(cases);
    return coooda_core::bench::run_cases(argc, argv, cases);
}
