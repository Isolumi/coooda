#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/benchmark.hpp>

int main() {
    const auto result = coooda_core::bench::time_once("compare_vector_add", []() {
        (void)coooda_compare::ops::vector_add_matches_reference();
    });
    coooda_core::bench::print_result(result);
    return 0;
}
