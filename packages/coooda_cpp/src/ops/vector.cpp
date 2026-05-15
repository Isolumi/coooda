#include <coooda_cpp/ops/vector.hpp>
#include <coooda_core/status.hpp>

#include <cstddef>

namespace coooda_cpp::ops {

std::vector<float> vector_add_reference(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        coooda_core::fail("vector_add_reference requires equal input sizes");
    }

    std::vector<float> out(a.size(), 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

} // namespace coooda_cpp::ops
