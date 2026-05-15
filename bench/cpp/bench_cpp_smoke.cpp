#include <coooda_core/benchmark.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/vector.hpp>

#include <vector>

int main(int argc, char **argv) {
    return coooda_core::bench::run_cases(argc, argv, {
        {"smoke",
         "tiny C++ vector-add smoke benchmark",
         []() {
             const auto result = coooda_core::bench::time_once("cpp_vector_add_reference_smoke", []() {
                 (void)coooda_cpp::ops::vector_add_reference({1.0f, 2.0f}, {3.0f, 4.0f});
             });
             coooda_core::bench::print_result(result);
         }},
        {"vector_ops",
         "seeded C++ vector-add benchmark",
         []() {
             const std::vector<float> a = coooda_core::test::seeded_vector(4096, 0xC001U, -10.0f, 10.0f);
             const std::vector<float> b = coooda_core::test::seeded_vector(4096, 0xD00DU, -10.0f, 10.0f);
             const auto result = coooda_core::bench::time_once("cpp_vector_add_reference_vector_ops", [&]() {
                 (void)coooda_cpp::ops::vector_add_reference(a, b);
             });
             coooda_core::bench::print_result(result);
         }},
    });
}
