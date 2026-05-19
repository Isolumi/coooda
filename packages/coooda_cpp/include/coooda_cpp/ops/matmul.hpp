#pragma once

#include <coooda_core/tensor.hpp>

namespace coooda_cpp::ops {

coooda_core::Tensor matmul_reference(
    const coooda_core::Tensor &a,
    const coooda_core::Tensor &b
);

} // namespace coooda_cpp::ops
