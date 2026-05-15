#include <coooda_core/benchmark.hpp>
#include <coooda_cuda/ops/vector.cuh>

int main() {
    const auto result = coooda_core::bench::time_once("cuda_vector_add_baseline", []() {
        (void)coooda_cuda::ops::vector_add_baseline({1.0f, 2.0f}, {3.0f, 4.0f});
    });
    coooda_core::bench::print_result(result);
    return 0;
}
