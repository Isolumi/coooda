#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/benchmark.hpp>

int main(int argc, char **argv) {
    return coooda_core::bench::run_cases(argc, argv, {
        {"smoke",
         "compare backend vector-add smoke benchmark",
         []() {
             const auto result = coooda_core::bench::time_once("compare_vector_add_smoke", []() {
                 (void)coooda_compare::ops::vector_add_matches_reference();
             });
             coooda_core::bench::print_result(result);
         }},
        {"vector_ops",
         "compare backend seeded vector-add benchmark",
         []() {
             const auto result = coooda_core::bench::time_once("compare_vector_add_vector_ops", []() {
                 (void)coooda_compare::ops::vector_add_matches_reference();
             });
             coooda_core::bench::print_result(result);
         }},
    });
}
