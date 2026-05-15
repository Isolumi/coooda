#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_cpp/package.hpp>

#include <vector>

int main() {
    return coooda_core::test::run_tests({
        {"cpp_backend_name", []() {
             coooda_core::test::require_equal("coooda_cpp", coooda_cpp::backend_name(), "backend name");
         }},
        {"cpp_vector_add_reference", []() {
             const std::vector<float> actual =
                 coooda_cpp::ops::vector_add_reference({1.0f, 2.0f}, {3.0f, 4.0f});
             coooda_core::test::require(actual.size() == 2, "vector add size");
             coooda_core::test::require(actual[0] == 4.0f, "vector add element 0");
             coooda_core::test::require(actual[1] == 6.0f, "vector add element 1");
         }},
    });
}
