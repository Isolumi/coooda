#include <coooda_core/test_harness.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace coooda_core::test {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(const std::string &expected, const std::string &actual, const std::string &message) {
    if (expected != actual) {
        throw std::runtime_error(message + ": expected '" + expected + "', got '" + actual + "'");
    }
}

int run_tests(const std::vector<TestCase> &tests) {
    int failures = 0;
    for (const TestCase &test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << "\n";
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}

bool close_enough(float expected, float actual, float abs_tol, float rel_tol) {
    if (expected == actual) {
        return true;
    }

    const float difference = std::fabs(expected - actual);
    const float scale = std::max(std::fabs(expected), std::fabs(actual));
    const float tolerance = abs_tol + (rel_tol * scale);
    return difference <= tolerance;
}

std::string MismatchReport::to_string() const {
    std::ostringstream out;
    if (matched) {
        out << label << ": matched";
        return out.str();
    }

    out << label << ": mismatch at index " << index << " expected " << expected << ", actual "
        << actual << ", absolute difference " << absolute_difference << ", tolerance " << tolerance;
    return out.str();
}

MismatchReport compare_vectors(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    float abs_tol,
    float rel_tol
) {
    const std::size_t common_size = std::min(expected.size(), actual.size());
    for (std::size_t i = 0; i < common_size; ++i) {
        const float difference = std::fabs(expected[i] - actual[i]);
        const float scale = std::max(std::fabs(expected[i]), std::fabs(actual[i]));
        const float tolerance = abs_tol + (rel_tol * scale);
        if (!close_enough(expected[i], actual[i], abs_tol, rel_tol)) {
            return MismatchReport{label, i, expected[i], actual[i], difference, tolerance, false};
        }
    }

    if (expected.size() != actual.size()) {
        const std::size_t index = common_size;
        const float expected_value =
            index < expected.size() ? expected[index] : std::numeric_limits<float>::quiet_NaN();
        const float actual_value =
            index < actual.size() ? actual[index] : std::numeric_limits<float>::quiet_NaN();
        return MismatchReport{
            label,
            index,
            expected_value,
            actual_value,
            std::numeric_limits<float>::infinity(),
            0.0f,
            false
        };
    }

    return MismatchReport{label, 0, 0.0f, 0.0f, 0.0f, 0.0f, true};
}

std::vector<float> seeded_vector(std::size_t size, std::uint32_t seed, float min_value, float max_value) {
    if (min_value > max_value) {
        throw std::invalid_argument("seeded_vector requires min_value <= max_value");
    }

    std::uint32_t state = seed == 0 ? 0x6d2b79f5U : seed;
    std::vector<float> values(size, 0.0f);
    for (float &value : values) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        const float unit = static_cast<float>(state) /
                           static_cast<float>(std::numeric_limits<std::uint32_t>::max());
        value = min_value + ((max_value - min_value) * unit);
    }
    return values;
}

} // namespace coooda_core::test
