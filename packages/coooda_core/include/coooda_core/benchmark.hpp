#pragma once

#include <functional>
#include <string>
#include <vector>

namespace coooda_core::bench {

struct BenchmarkResult {
    std::string name;
    double elapsed_ms;
};

struct BenchmarkCase {
    std::string name;
    std::string description;
    std::function<void()> run;
};

BenchmarkResult time_once(const std::string &name, const std::function<void()> &fn);
void print_result(const BenchmarkResult &result);
int run_cases(int argc, char **argv, const std::vector<BenchmarkCase> &cases);

} // namespace coooda_core::bench
