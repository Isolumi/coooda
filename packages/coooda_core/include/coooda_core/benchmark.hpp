#pragma once

#include <functional>
#include <string>

namespace coooda_core::bench {

struct BenchmarkResult {
    std::string name;
    double elapsed_ms;
};

BenchmarkResult time_once(const std::string &name, const std::function<void()> &fn);
void print_result(const BenchmarkResult &result);

} // namespace coooda_core::bench
