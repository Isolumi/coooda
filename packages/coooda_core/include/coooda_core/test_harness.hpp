#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace coooda_core::test {

using TestFn = std::function<void()>;

struct TestCase {
    std::string name;
    TestFn run;
};

void require(bool condition, const std::string &message);
void require_equal(const std::string &expected, const std::string &actual, const std::string &message);
int run_tests(const std::vector<TestCase> &tests);

bool close_enough(float expected, float actual, float abs_tol, float rel_tol);

struct MismatchReport {
    std::string label;
    std::size_t index = 0;
    float expected = 0.0f;
    float actual = 0.0f;
    float absolute_difference = 0.0f;
    float tolerance = 0.0f;
    bool matched = true;

    std::string to_string() const;
};

MismatchReport compare_vectors(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    float abs_tol,
    float rel_tol
);

std::vector<float> seeded_vector(std::size_t size, std::uint32_t seed, float min_value, float max_value);

} // namespace coooda_core::test
