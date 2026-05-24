#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/benchmark.hpp>

#include <vector>

namespace coooda_bench::compare {

void append_core_helper_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_embeddings_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_elementwise_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_matmul_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_memory_tensor_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_normalization_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_reductions_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_softmax_loss_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);
void append_vector_ops_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases);

} // namespace coooda_bench::compare

int main(int argc, char **argv) {
    std::vector<coooda_core::bench::BenchmarkCase> cases{
        {"smoke",
         "compare backend vector-add smoke benchmark",
         []() {
             const auto result = coooda_core::bench::time_once("compare_vector_add_smoke", []() {
                 (void)coooda_compare::ops::vector_add_matches_reference();
             });
             coooda_core::bench::print_result(result);
         }},
    };

    coooda_bench::compare::append_core_helper_cases(cases);
    coooda_bench::compare::append_embeddings_compare_cases(cases);
    coooda_bench::compare::append_elementwise_compare_cases(cases);
    coooda_bench::compare::append_matmul_compare_cases(cases);
    coooda_bench::compare::append_memory_tensor_cases(cases);
    coooda_bench::compare::append_normalization_compare_cases(cases);
    coooda_bench::compare::append_reductions_compare_cases(cases);
    coooda_bench::compare::append_softmax_loss_compare_cases(cases);
    coooda_bench::compare::append_vector_ops_compare_cases(cases);
    return coooda_core::bench::run_cases(argc, argv, cases);
}
