#include <coooda_core/benchmark.hpp>

#include <chrono>
#include <iostream>

namespace coooda_core::bench {

BenchmarkResult time_once(const std::string &name, const std::function<void()> &fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return BenchmarkResult{name, elapsed.count()};
}

void print_result(const BenchmarkResult &result) {
    std::cout << result.name << ": " << result.elapsed_ms << " ms\n";
}

} // namespace coooda_core::bench
