#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#ifndef COODA_BUILD_MODE
#define COODA_BUILD_MODE "unknown"
#endif

namespace coooda::bench {

struct BenchmarkConfig {
    std::string operation;
    std::string variant;
    std::string shape;
    std::string dtype;
    std::string build_mode = COODA_BUILD_MODE;
    std::string gpu = "unknown";
    std::size_t warmups = 5;
    std::size_t trials = 20;
    bool synchronize_before_each_trial = true;
    bool synchronize_after_each_trial = true;
};

struct TimingStats {
    std::string operation;
    std::string variant;
    std::string shape;
    std::string dtype;
    std::string build_mode;
    std::string gpu;
    std::size_t warmups = 0;
    std::size_t trials = 0;
    std::vector<double> samples_ms;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;

    std::string to_string() const;
};

using BenchmarkFn = std::function<void()>;

TimingStats run_benchmark(const BenchmarkConfig &config, const BenchmarkFn &benchmark);

std::ostream &operator<<(std::ostream &os, const TimingStats &stats);

} // namespace coooda::bench
