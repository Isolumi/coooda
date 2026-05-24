#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/embedding.hpp>
#include <coooda_cuda/nn/embedding.cuh>

#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-5f;
constexpr float kRelTol = 1.0e-5f;

struct EmbeddingInputs {
    std::vector<std::size_t> ids;
    std::vector<float> token_table;
    std::vector<float> position_table;
    std::size_t embedding_dim = 0;
};

struct EmbeddingExpected {
    std::vector<float> lookup;
    std::vector<float> token_position;
};

std::vector<std::size_t> cycling_ids(std::size_t count, std::size_t vocab_size) {
    std::vector<std::size_t> ids(count, 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = (i * 17) % vocab_size;
    }
    return ids;
}

EmbeddingInputs make_inputs() {
    constexpr std::size_t sequence_length = 4096;
    constexpr std::size_t vocab_size = 8192;
    constexpr std::size_t embedding_dim = 256;

    return EmbeddingInputs{
        cycling_ids(sequence_length, vocab_size),
        coooda_core::test::seeded_vector(vocab_size * embedding_dim, 0x1001U, -2.0f, 2.0f),
        coooda_core::test::seeded_vector(sequence_length * embedding_dim, 0x1002U, -0.5f, 0.5f),
        embedding_dim,
    };
}

EmbeddingExpected make_expected(const EmbeddingInputs &inputs) {
    return EmbeddingExpected{
        coooda_cpp::nn::embedding_lookup_reference(
            inputs.ids,
            inputs.token_table,
            inputs.embedding_dim
        ),
        coooda_cpp::nn::token_position_embedding_reference(
            inputs.ids,
            inputs.token_table,
            inputs.position_table,
            inputs.embedding_dim
        ),
    };
}

void require_vector_close(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    if (!report.matched) {
        coooda_core::fail(report.to_string());
    }
}

void warm_up_cuda(const EmbeddingInputs &inputs) {
    (void)coooda_cuda::nn::embedding_lookup_baseline(
        inputs.ids,
        inputs.token_table,
        inputs.embedding_dim
    );
    (void)coooda_cuda::nn::embedding_lookup_grid_stride(
        inputs.ids,
        inputs.token_table,
        inputs.embedding_dim
    );
    (void)coooda_cuda::nn::token_position_embedding_baseline(
        inputs.ids,
        inputs.token_table,
        inputs.position_table,
        inputs.embedding_dim
    );
    (void)coooda_cuda::nn::token_position_embedding_grid_stride(
        inputs.ids,
        inputs.token_table,
        inputs.position_table,
        inputs.embedding_dim
    );
}

void run_embeddings_compare_case() {
    const EmbeddingInputs inputs = make_inputs();
    const EmbeddingExpected expected = make_expected(inputs);
    warm_up_cuda(inputs);

    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_embeddings_lookup_baseline", [&]() {
        require_vector_close(
            "compare_embeddings_lookup_baseline",
            expected.lookup,
            coooda_cuda::nn::embedding_lookup_baseline(
                inputs.ids,
                inputs.token_table,
                inputs.embedding_dim
            )
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_embeddings_lookup_grid_stride", [&]() {
        require_vector_close(
            "compare_embeddings_lookup_grid_stride",
            expected.lookup,
            coooda_cuda::nn::embedding_lookup_grid_stride(
                inputs.ids,
                inputs.token_table,
                inputs.embedding_dim
            )
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_embeddings_token_position_baseline", [&]() {
        require_vector_close(
            "compare_embeddings_token_position_baseline",
            expected.token_position,
            coooda_cuda::nn::token_position_embedding_baseline(
                inputs.ids,
                inputs.token_table,
                inputs.position_table,
                inputs.embedding_dim
            )
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_embeddings_token_position_grid_stride", [&]() {
        require_vector_close(
            "compare_embeddings_token_position_grid_stride",
            expected.token_position,
            coooda_cuda::nn::token_position_embedding_grid_stride(
                inputs.ids,
                inputs.token_table,
                inputs.position_table,
                inputs.embedding_dim
            )
        );
    }));
}

} // namespace

namespace coooda_bench::compare {

void append_embeddings_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "embeddings",
        "compare C++ reference, CUDA baseline, and CUDA grid-stride embedding lookup paths",
        []() { run_embeddings_compare_case(); },
    });
}

} // namespace coooda_bench::compare
