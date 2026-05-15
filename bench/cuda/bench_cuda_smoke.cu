#include <coooda_core/benchmark.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <vector>

int main(int argc, char **argv) {
    return coooda_core::bench::run_cases(argc, argv, {
        {"smoke",
         "tiny CUDA vector-add smoke benchmark",
         []() {
             const auto result = coooda_core::bench::time_once("cuda_vector_add_baseline_smoke", []() {
                 (void)coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f}, {3.0f, 4.0f});
             });
             coooda_core::bench::print_result(result);
         }},
        {"vector_ops",
         "seeded CUDA vector-add benchmark",
         []() {
             const std::vector<float> a = coooda_core::test::seeded_vector(4096, 0xC001U, -10.0f, 10.0f);
             const std::vector<float> b = coooda_core::test::seeded_vector(4096, 0xD00DU, -10.0f, 10.0f);
             const auto result = coooda_core::bench::time_once("cuda_vector_add_baseline_vector_ops", [&]() {
                 (void)coooda_cuda::ops::vector_add_baseline(a, b);
             });
             coooda_core::bench::print_result(result);
         }},
    });
}
