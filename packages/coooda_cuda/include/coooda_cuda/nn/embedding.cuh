#pragma once

#include <cstddef>
#include <vector>

namespace coooda_cuda::nn {

std::vector<float> embedding_lookup_baseline(
    const std::vector<std::size_t> &ids,
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim
);

std::vector<float> embedding_lookup_grid_stride(
    const std::vector<std::size_t> &ids,
    const std::vector<float> &embedding_table,
    std::size_t embedding_dim
);

std::vector<float> token_position_embedding_baseline(
    const std::vector<std::size_t> &token_ids,
    const std::vector<float> &token_embedding_table,
    const std::vector<float> &position_embedding_table,
    std::size_t embedding_dim
);

std::vector<float> token_position_embedding_grid_stride(
    const std::vector<std::size_t> &token_ids,
    const std::vector<float> &token_embedding_table,
    const std::vector<float> &position_embedding_table,
    std::size_t embedding_dim
);

} // namespace coooda_cuda::nn
