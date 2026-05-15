#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace coooda_core {

struct Shape {
    std::vector<std::size_t> dims;
};

std::size_t numel(const Shape &shape);
std::string to_string(const Shape &shape);

} // namespace coooda_core
