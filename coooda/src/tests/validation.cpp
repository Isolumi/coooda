#include <coooda/tests/validation.hpp>

#include <algorithm>
#include <cmath>
#include <ostream>
#include <sstream>

namespace coooda::tests {

bool close_enough(
    double actual,
    double expected,
    double absolute_tolerance,
    double relative_tolerance
) {
    if (actual == expected) {
        return true;
    }
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return false;
    }

    const double difference = std::abs(actual - expected);
    const double tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);
    return difference <= tolerance;
}

std::string FailureReport::to_string() const {
    std::ostringstream os;
    os << "operation=" << operation << ", shape=" << shape << ", seed=" << seed
       << ", first_mismatch=" << first_mismatch << ", expected=" << expected_value
       << ", actual=" << actual_value << ", absolute_difference=" << absolute_difference
       << ", relative_difference=" << relative_difference << ", tolerance=" << tolerance
       << ", maximum_observed_error=" << maximum_observed_error;
    return os.str();
}

std::ostream &operator<<(
    std::ostream &os,
    const FailureReport &report
) {
    os << report.to_string();
    return os;
}

} // namespace coooda::tests
