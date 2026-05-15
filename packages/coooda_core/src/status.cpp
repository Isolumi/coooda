#include <coooda_core/status.hpp>

namespace coooda_core {

Error::Error(const std::string &message) : std::runtime_error(message) {}

void fail(const std::string &message) {
    throw Error(message);
}

} // namespace coooda_core
