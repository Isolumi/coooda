#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/embedding.hpp>
#include <coooda_cuda/nn/embedding.cuh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-5f;
constexpr float kRelTol = 1.0e-5f;

void require_vector_close(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

void compare_case(
    const std::string &label,
    const std::vector<std::size_t> &ids,
    const std::vector<float> &token_table,
    const std::vector<float> &position_table,
    std::size_t embedding_dim
) {
    const std::vector<float> lookup_expected =
        coooda_cpp::nn::embedding_lookup_reference(ids, token_table, embedding_dim);
    require_vector_close(
        label + "_lookup_baseline",
        lookup_expected,
        coooda_cuda::nn::embedding_lookup_baseline(ids, token_table, embedding_dim)
    );
    require_vector_close(
        label + "_lookup_grid_stride",
        lookup_expected,
        coooda_cuda::nn::embedding_lookup_grid_stride(ids, token_table, embedding_dim)
    );

    const std::vector<float> combined_expected =
        coooda_cpp::nn::token_position_embedding_reference(ids, token_table, position_table, embedding_dim);
    require_vector_close(
        label + "_token_position_baseline",
        combined_expected,
        coooda_cuda::nn::token_position_embedding_baseline(ids, token_table, position_table, embedding_dim)
    );
    require_vector_close(
        label + "_token_position_grid_stride",
        combined_expected,
        coooda_cuda::nn::token_position_embedding_grid_stride(ids, token_table, position_table, embedding_dim)
    );
}

std::vector<std::size_t> cycling_ids(std::size_t count, std::size_t vocab_size) {
    std::vector<std::size_t> ids(count, 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = (i * 7) % vocab_size;
    }
    return ids;
}

std::vector<float> seeded(std::size_t size, std::uint32_t seed, float low = -2.0f, float high = 2.0f) {
    return coooda_core::test::seeded_vector(size, seed, low, high);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"embeddings_compare_known_values", []() {
             compare_case(
                 "known",
                 {2, 0, 1},
                 {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f},
                 {0.5f, -0.5f, 2.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                 3
             );
         }},

        {"embeddings_compare_repeated_and_boundary_ids", []() {
             compare_case(
                 "boundary",
                 {3, 0, 3, 1},
                 {0.0f, 1.0f, 10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f},
                 {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f},
                 2
             );
         }},

        {"embeddings_compare_seeded_small", []() {
             compare_case(
                 "seeded_small",
                 cycling_ids(17, 13),
                 seeded(13 * 8, 0x1001U),
                 seeded(17 * 8, 0x1002U, -0.5f, 0.5f),
                 8
             );
         }},

        {"embeddings_compare_seeded_large", []() {
             compare_case(
                 "seeded_large",
                 cycling_ids(513, 257),
                 seeded(257 * 64, 0x1003U),
                 seeded(513 * 64, 0x1004U, -0.5f, 0.5f),
                 64
             );
         }},
    });
}
