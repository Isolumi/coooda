#include <coooda/bench/benchmark.hpp>

#include <coooda/core/cuda_checks.hpp>
#include <coooda/core/timers.hpp>

#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace coooda::bench {
namespace {

double median_of_sorted(
    const std::vector<double> &sorted_samples
) {
    const std::size_t count = sorted_samples.size();
    const std::size_t middle = count / 2;
    if (count % 2 == 1) {
        return sorted_samples[middle];
    }
    return 0.5 * (sorted_samples[middle - 1] + sorted_samples[middle]);
}

} // namespace

TimingStats run_benchmark(
    const BenchmarkConfig &config,
    const BenchmarkFn &benchmark
) {
    if (!benchmark) {
        throw std::invalid_argument("run_benchmark requires a callable");
    }
    if (config.trials == 0) {
        throw std::invalid_argument("run_benchmark requires at least one trial");
    }

    for (std::size_t warmup = 0; warmup < config.warmups; ++warmup) {
        benchmark();
        if (config.synchronize_after_each_trial) {
            COODA_SYNCHRONIZE_DEVICE();
        }
    }

    TimingStats stats;
    stats.operation = config.operation;
    stats.variant = config.variant;
    stats.shape = config.shape;
    stats.dtype = config.dtype;
    stats.build_mode = config.build_mode;
    stats.gpu = config.gpu;
    stats.warmups = config.warmups;
    stats.trials = config.trials;
    stats.samples_ms.reserve(config.trials);

    for (std::size_t trial = 0; trial < config.trials; ++trial) {
        if (config.synchronize_before_each_trial) {
            COODA_SYNCHRONIZE_DEVICE();
        }

        coooda::core::CpuTimer timer;
        timer.start();
        benchmark();
        if (config.synchronize_after_each_trial) {
            COODA_SYNCHRONIZE_DEVICE();
        }
        stats.samples_ms.push_back(timer.stop_ms());
    }

    std::vector<double> sorted_samples = stats.samples_ms;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    stats.min_ms = sorted_samples.front();
    stats.max_ms = sorted_samples.back();
    stats.median_ms = median_of_sorted(sorted_samples);
    stats.mean_ms = std::accumulate(sorted_samples.begin(), sorted_samples.end(), 0.0) /
                    static_cast<double>(sorted_samples.size());

    return stats;
}

std::string TimingStats::to_string() const {
    std::ostringstream os;
    os << "operation=" << operation << ", variant=" << variant << ", shape=" << shape
       << ", dtype=" << dtype << ", build_mode=" << build_mode << ", gpu=" << gpu
       << ", warmups=" << warmups << ", trials=" << trials << ", median_ms=" << median_ms
       << ", min_ms=" << min_ms << ", max_ms=" << max_ms << ", mean_ms=" << mean_ms;
    return os.str();
}

std::ostream &operator<<(
    std::ostream &os,
    const TimingStats &stats
) {
    os << stats.to_string();
    return os;
}

} // namespace coooda::bench
