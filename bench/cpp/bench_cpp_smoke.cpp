#include <coooda_core/benchmark.hpp>
#include <coooda_cpp/ops/vector.hpp>

int main() {
    const auto result = coooda_core::bench::time_once("cpp_vector_add_reference", []() {
        (void)coooda_cpp::ops::vector_add_reference({1.0f, 2.0f}, {3.0f, 4.0f});
    });
    coooda_core::bench::print_result(result);
    return 0;
}
