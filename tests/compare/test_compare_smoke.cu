#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_compare/package.hpp>
#include <coooda_core/test_harness.hpp>

int main() {
    return coooda_core::test::run_tests({
        {"compare_package_name", []() {
             coooda_core::test::require_equal(
                 "coooda_compare", coooda_compare::package_name(), "package name"
             );
         }},
        {"compare_vector_add", []() {
             coooda_core::test::require(
                 coooda_compare::ops::vector_add_matches_reference(),
                 "CUDA vector add should match C++ reference"
             );
         }},
    });
}
