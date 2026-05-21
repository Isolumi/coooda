#pragma once

#include <coooda_core/tensor.hpp>

namespace coooda_cuda::ops {

coooda_core::Tensor matmul_baseline(
    const coooda_core::Tensor &a,
    const coooda_core::Tensor &b
);

coooda_core::Tensor matmul_tiled(
    const coooda_core::Tensor &a,
    const coooda_core::Tensor &b
);

} // namespace coooda_cuda::ops
