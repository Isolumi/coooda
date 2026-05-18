#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/test_harness.hpp>

int main() {
    return coooda_core::test::run_tests({
        {"vector_ops_compare_add", []() {
             coooda_core::test::require(
                 coooda_compare::ops::vector_add_matches_reference(),
                 "CUDA vector add should match C++ reference"
             );
         }},

        {"vector_ops_compare_saxpy", []() {
             coooda_core::test::require(
                 coooda_compare::ops::saxpy_matches_reference(),
                 "CUDA SAXPY should match C++ reference"
             );
         }},

        {"vector_ops_compare_relu", []() {
             coooda_core::test::require(
                 coooda_compare::ops::relu_matches_reference(),
                 "CUDA ReLU should match C++ reference"
             );
         }},

        {"vector_ops_compare_sigmoid", []() {
             coooda_core::test::require(
                 coooda_compare::ops::sigmoid_matches_reference(),
                 "CUDA sigmoid should match C++ reference"
             );
         }},

        {"vector_ops_compare_elementwise_multiply", []() {
             coooda_core::test::require(
                 coooda_compare::ops::elementwise_multiply_matches_reference(),
                 "CUDA elementwise multiply should match C++ reference"
             );
         }},

        {"vector_ops_compare_all", []() {
             coooda_core::test::require(
                 coooda_compare::ops::vector_ops_match_reference(),
                 "all CUDA vector ops should match C++ references"
             );
         }},
    });
}
