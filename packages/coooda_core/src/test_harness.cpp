#include <coooda_core/test_harness.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>

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

} // namespace coooda_core::test
