#include <coooda_core/tensor.hpp>

#include <numeric>
#include <sstream>

namespace coooda_core {

std::size_t numel(const Shape &shape) {
    if (shape.dims.empty()) {
        return 0;
    }

    return std::accumulate(
        shape.dims.begin(),
        shape.dims.end(),
        static_cast<std::size_t>(1),
        [](std::size_t acc, std::size_t dim) { return acc * dim; }
    );
}

std::string to_string(const Shape &shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.dims.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        out << shape.dims[i];
    }
    out << "]";
    return out.str();
}

} // namespace coooda_core
