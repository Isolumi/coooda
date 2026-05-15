#pragma once

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

} // namespace coooda_core::test
