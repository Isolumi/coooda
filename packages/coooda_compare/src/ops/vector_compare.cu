#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <cmath>
#include <cstddef>
#include <vector>

namespace coooda_compare::ops {

bool vector_add_matches_reference() {
    const std::vector<float> a{1.0f, -2.0f, 3.5f, 4.0f};
    const std::vector<float> b{0.5f, 2.0f, -1.5f, 8.0f};
    const std::vector<float> expected = coooda_cpp::ops::vector_add_reference(a, b);
    const std::vector<float> actual = coooda_cuda::ops::vector_add_baseline(a, b);

    if (expected.size() != actual.size()) {
        return false;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > 1.0e-6f) {
            return false;
        }
    }
    return true;
}

} // namespace coooda_compare::ops
