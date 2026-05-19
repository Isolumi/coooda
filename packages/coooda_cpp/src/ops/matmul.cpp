#include <coooda_cpp/ops/matmul.hpp>

#include <coooda_core/status.hpp>

#include <cstddef>
#include <string>

namespace {

void require_matrix(const coooda_core::Tensor &tensor, const char *name) {
    if (tensor.shape().dims.size() != 2) {
        coooda_core::fail(std::string("matmul_reference requires ") + name + " to be rank 2");
    }
}

std::size_t rows(const coooda_core::Tensor &tensor) {
    return tensor.shape().dims[0];
}

std::size_t cols(const coooda_core::Tensor &tensor) {
    return tensor.shape().dims[1];
}

} // namespace

namespace coooda_cpp::ops {

coooda_core::Tensor matmul_reference(
    const coooda_core::Tensor &a,
    const coooda_core::Tensor &b
) {
    require_matrix(a, "left input");
    require_matrix(b, "right input");

    const std::size_t m = rows(a);
    const std::size_t k = cols(a);
    const std::size_t b_rows = rows(b);
    const std::size_t n = cols(b);

    if (k != b_rows) {
        coooda_core::fail("matmul_reference requires left columns to equal right rows");
    }

    coooda_core::Tensor output({{m, n}});
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float total = 0.0f;
            for (std::size_t inner = 0; inner < k; ++inner) {
                total += a[(row * k) + inner] * b[(inner * n) + col];
            }
            output[(row * n) + col] = total;
        }
    }

    return output;
}

} // namespace coooda_cpp::ops
