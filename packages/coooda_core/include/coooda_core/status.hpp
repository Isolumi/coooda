#pragma once

#include <stdexcept>
#include <string>

namespace coooda_core {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &message);
};

[[noreturn]] void fail(const std::string &message);

} // namespace coooda_core
