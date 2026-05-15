#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace coooda::tests {

bool close_enough(
    double actual,
    double expected,
    double absolute_tolerance = 1.0e-5,
    double relative_tolerance = 1.0e-5
);

struct FailureReport {
    std::string operation;
    std::string shape;
    std::uint64_t seed = 0;
    std::size_t first_mismatch = 0;
    double expected_value = 0.0;
    double actual_value = 0.0;
    double absolute_difference = 0.0;
    double relative_difference = 0.0;
    double tolerance = 0.0;
    double maximum_observed_error = 0.0;

    std::string to_string() const;
};

std::ostream &operator<<(std::ostream &os, const FailureReport &report);

} // namespace coooda::tests
