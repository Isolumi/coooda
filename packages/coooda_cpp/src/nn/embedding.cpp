#include <coooda_cpp/nn/embedding.hpp>

#include <coooda_core/status.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace {

std::size_t require_embedding_table_shape(
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim,
    const char *operation
) {
    if (embedding_dim == 0) {
        coooda_core::fail(std::string(operation) + " requires non-zero embedding dim");
    }
    if (embedding_table.size() % embedding_dim != 0) {
        coooda_core::fail(std::string(operation) + " embedding table size must be divisible by embedding dim");
    }
    return embedding_table.size() / embedding_dim;
}

void require_ids_in_range(
    const std::vector<std::size_t> &ids,
    std::size_t row_count,
    const char *operation
) {
    for (const std::size_t id : ids) {
        if (id >= row_count) {
            coooda_core::fail(std::string(operation) + " id is out of range");
        }
    }
}

} // namespace

namespace coooda_cpp::nn {

std::vector<float> embedding_lookup_reference(
    const std::vector<std::size_t> &ids,
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim
) {
    const std::size_t row_count =
        require_embedding_table_shape(embedding_table, embedding_dim, "embedding_lookup_reference");
    require_ids_in_range(ids, row_count, "embedding_lookup_reference");

    std::vector<float> out(ids.size() * embedding_dim, 0.0f);
    for (std::size_t token_index = 0; token_index < ids.size(); ++token_index) {
        const std::size_t table_offset = ids[token_index] * embedding_dim;
        const std::size_t out_offset = token_index * embedding_dim;
        for (std::size_t dim = 0; dim < embedding_dim; ++dim) {
            out[out_offset + dim] = embedding_table[table_offset + dim];
        }
    }
    return out;
}

std::vector<float> token_position_embedding_reference(
    const std::vector<std::size_t> &token_ids,
    const std::vector<float> &token_embedding_table,
    const std::vector<float> &position_embedding_table,
    std::size_t embedding_dim
) {
    const std::size_t vocab_size =
        require_embedding_table_shape(token_embedding_table, embedding_dim, "token_position_embedding_reference");
    const std::size_t position_count =
        require_embedding_table_shape(position_embedding_table, embedding_dim, "token_position_embedding_reference");
    require_ids_in_range(token_ids, vocab_size, "token_position_embedding_reference");
    if (token_ids.size() > position_count) {
        coooda_core::fail("token_position_embedding_reference position table is too short");
    }

    std::vector<float> out(token_ids.size() * embedding_dim, 0.0f);
    for (std::size_t token_index = 0; token_index < token_ids.size(); ++token_index) {
        const std::size_t token_offset = token_ids[token_index] * embedding_dim;
        const std::size_t position_offset = token_index * embedding_dim;
        const std::size_t out_offset = token_index * embedding_dim;
        for (std::size_t dim = 0; dim < embedding_dim; ++dim) {
            out[out_offset + dim] =
                token_embedding_table[token_offset + dim] + position_embedding_table[position_offset + dim];
        }
    }
    return out;
}

} // namespace coooda_cpp::nn
