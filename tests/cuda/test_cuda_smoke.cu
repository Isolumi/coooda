#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/vector.cuh>
#include <coooda_cuda/package.cuh>

#include <vector>

int main() {
    return coooda_core::test::run_tests({
        {"cuda_backend_name", []() {
             coooda_core::test::require_equal("coooda_cuda", coooda_cuda::backend_name(), "backend name");
         }},
        {"cuda_vector_add_baseline", []() {
             const std::vector<float> actual =
                 coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f}, {3.0f, 4.0f});
             coooda_core::test::require(actual.size() == 2, "vector add size");
             coooda_core::test::require(actual[0] == 4.0f, "vector add element 0");
             coooda_core::test::require(actual[1] == 6.0f, "vector add element 1");
         }},
    });
}
