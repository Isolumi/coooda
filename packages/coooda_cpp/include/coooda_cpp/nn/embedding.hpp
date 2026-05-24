#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cpp::nn {

std::vector<float> embedding_lookup_reference(
    const std::vector<std::size_t> &ids,
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim
);

std::vector<float> token_position_embedding_reference(
    const std::vector<std::size_t> &token_ids,
    const std::vector<float> &token_embedding_table,
    const std::vector<float> &position_embedding_table,
    std::size_t embedding_dim
);

} // namespace coooda_cpp::nn
