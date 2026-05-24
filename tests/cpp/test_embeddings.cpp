#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/embedding.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-5f;
constexpr float kRelTol = 1.0e-5f;

void require_vector_close(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const std::string &label
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

template <typename Fn>
void require_core_error(Fn &&fn, const std::string &message) {
    bool threw = false;
    try {
        fn();
    } catch (const coooda_core::Error &) {
        threw = true;
    }
    coooda_core::test::require(threw, message);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"embeddings_cpp_lookup_known_values", []() {
             require_vector_close(
                 {20.0f, 21.0f, 22.0f, 0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f},
                 coooda_cpp::nn::embedding_lookup_reference(
                     {2, 0, 1},
                     {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f},
                     3
                 ),
                 "embedding lookup known values"
             );
         }},

        {"embeddings_cpp_token_position_known_values", []() {
             require_vector_close(
                 {20.5f, 20.5f, 24.0f, 4.0f, 6.0f, 8.0f},
                 coooda_cpp::nn::token_position_embedding_reference(
                     {2, 0},
                     {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f},
                     {0.5f, -0.5f, 2.0f, 4.0f, 5.0f, 6.0f},
                     3
                 ),
                 "token position embedding known values"
             );
         }},

        {"embeddings_cpp_repeated_and_boundary_ids", []() {
             require_vector_close(
                 {30.0f, 31.0f, 0.0f, 1.0f, 30.0f, 31.0f},
                 coooda_cpp::nn::embedding_lookup_reference(
                     {3, 0, 3},
                     {0.0f, 1.0f, 10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f},
                     2
                 ),
                 "embedding boundary ids"
             );
         }},

        {"embeddings_cpp_empty_tokens", []() {
             coooda_core::test::require(
                 coooda_cpp::nn::embedding_lookup_reference({}, {1.0f, 2.0f, 3.0f, 4.0f}, 2).empty(),
                 "empty lookup"
             );
             coooda_core::test::require(
                 coooda_cpp::nn::token_position_embedding_reference({}, {1.0f, 2.0f}, {}, 2).empty(),
                 "empty token position embedding"
             );
         }},

        {"embeddings_cpp_seeded_output_is_finite", []() {
             const std::vector<float> token_table =
                 coooda_core::test::seeded_vector(8 * 6, 0x1001U, -2.0f, 2.0f);
             const std::vector<float> position_table =
                 coooda_core::test::seeded_vector(4 * 6, 0x1002U, -0.5f, 0.5f);
             const std::vector<float> out =
                 coooda_cpp::nn::token_position_embedding_reference({7, 3, 0, 6}, token_table, position_table, 6);

             coooda_core::test::require(out.size() == 24, "seeded embedding output size");
             for (const float value : out) {
                 coooda_core::test::require(std::isfinite(value), "embedding output should stay finite");
             }
         }},

        {"embeddings_cpp_rejects_invalid_inputs", []() {
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::embedding_lookup_reference({0}, {1.0f, 2.0f, 3.0f}, 2);
                 },
                 "lookup should reject table not divisible by dim"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::embedding_lookup_reference({2}, {1.0f, 2.0f, 3.0f, 4.0f}, 2);
                 },
                 "lookup should reject out-of-range token id"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::token_position_embedding_reference(
                         {0, 1, 0},
                         {1.0f, 2.0f, 3.0f, 4.0f},
                         {0.0f, 0.0f, 1.0f, 1.0f},
                         2
                     );
                 },
                 "token position embedding should reject short position table"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::embedding_lookup_reference({}, {}, 0);
                 },
                 "lookup should reject zero embedding dim"
             );
         }},
    });
}
