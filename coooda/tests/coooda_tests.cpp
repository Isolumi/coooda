#include <coooda/bench/benchmark.hpp>
#include <coooda/core/cuda_checks.hpp>
#include <coooda/core/device.hpp>
#include <coooda/core/timers.hpp>
#include <coooda/tests/validation.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using TestFn = void (*)();

struct TestCase {
    const char *name;
    TestFn run;
};

void require(
    bool condition,
    const std::string &message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_contains(
    const std::string &text,
    const std::string &expected
) {
    require(
        text.find(expected) != std::string::npos,
        "expected text to contain '" + expected + "', got '" + text + "'"
    );
}

void test_close_enough_uses_absolute_and_relative_tolerance() {
    using coooda::tests::close_enough;

    require(
        close_enough(1.0, 1.0 + 1.0e-7, 1.0e-6, 1.0e-6),
        "absolute tolerance should accept tiny differences"
    );
    require(
        close_enough(1000.0, 1000.5, 1.0e-6, 1.0e-3),
        "relative tolerance should scale with expected value"
    );
    require(!close_enough(1.0, 1.1, 1.0e-6, 1.0e-6), "values outside tolerance should fail");
    require(
        !close_enough(std::nan(""), 1.0, 1.0e-6, 1.0e-6),
        "NaN should not compare close to a finite value"
    );
}

void test_failure_report_formats_validation_context() {
    coooda::tests::FailureReport report;
    report.operation = "add";
    report.shape = "4x4";
    report.seed = 1234;
    report.first_mismatch = 7;
    report.expected_value = 1.0;
    report.actual_value = 1.25;
    report.absolute_difference = 0.25;
    report.relative_difference = 0.25;
    report.tolerance = 1.0e-5;
    report.maximum_observed_error = 0.25;

    const std::string formatted = report.to_string();
    require_contains(formatted, "operation=add");
    require_contains(formatted, "shape=4x4");
    require_contains(formatted, "seed=1234");
    require_contains(formatted, "first_mismatch=7");
    require_contains(formatted, "expected=1");
    require_contains(formatted, "actual=1.25");
    require_contains(formatted, "absolute_difference=0.25");
    require_contains(formatted, "relative_difference=0.25");
    require_contains(formatted, "tolerance=1e-05");
    require_contains(formatted, "maximum_observed_error=0.25");
}

void test_check_cuda_reports_file_line_expression_and_message() {
    try {
        coooda::core::check_cuda(cudaErrorInvalidValue, "synthetic_call", "synthetic_file.cu", 77);
    } catch (const coooda::core::CudaError &error) {
        const std::string message = error.what();
        require(error.code() == cudaErrorInvalidValue, "CudaError should store code");
        require_contains(message, "synthetic_file.cu:77");
        require_contains(message, "synthetic_call");
        require_contains(message, "invalid argument");
        return;
    }

    throw std::runtime_error("check_cuda should throw for cudaErrorInvalidValue");
}

void test_cpu_timer_measures_elapsed_milliseconds() {
    coooda::core::CpuTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const double elapsed_ms = timer.stop_ms();

    require(elapsed_ms >= 0.0, "CPU timer should report non-negative time");
    require(!timer.running(), "CPU timer should not be running after stop");
}

void test_run_benchmark_records_warmups_trials_and_summary_stats() {
    coooda::bench::BenchmarkConfig config;
    config.operation = "counter";
    config.variant = "host";
    config.shape = "scalar";
    config.dtype = "int";
    config.build_mode = "test";
    config.gpu = "none";
    config.warmups = 2;
    config.trials = 5;
    config.synchronize_before_each_trial = false;
    config.synchronize_after_each_trial = false;

    int calls = 0;
    const coooda::bench::TimingStats stats =
        coooda::bench::run_benchmark(config, [&calls]() { ++calls; });

    require(calls == 7, "benchmark should run warmups plus trials");
    require(stats.warmups == 2, "stats should include warmup count");
    require(stats.trials == 5, "stats should include trial count");
    require(stats.samples_ms.size() == 5, "stats should keep trial samples");
    require(stats.min_ms <= stats.median_ms, "min should be <= median");
    require(stats.median_ms <= stats.max_ms, "median should be <= max");
    require(stats.operation == "counter", "stats should copy operation metadata");
    require(stats.variant == "host", "stats should copy variant metadata");
}

void test_cuda_runtime_helpers_when_device_is_available() {
    try {
        const int count = coooda::core::device_count();
        if (count == 0) {
            std::cout << "Skipping CUDA device runtime checks: no CUDA devices\n";
            return;
        }

        const coooda::core::DeviceInfo info = coooda::core::get_device_info(0);
        require(info.id == 0, "device info id should match requested device");
        require(!info.name.empty(), "device info should include name");
        require(info.global_memory_bytes > 0, "device info should include global memory");
        require(info.warp_size > 0, "device info should include warp size");

        coooda::core::CudaEventTimer timer;
        timer.start();
        coooda::core::synchronize_device();
        const float elapsed_ms = timer.stop_ms();
        require(elapsed_ms >= 0.0f, "CUDA event timer should be non-negative");
    } catch (const coooda::core::CudaError &error) {
        std::cout << "Skipping CUDA device runtime checks: " << error.what() << "\n";
    }
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"close_enough", test_close_enough_uses_absolute_and_relative_tolerance},
        {"failure_report", test_failure_report_formats_validation_context},
        {"check_cuda", test_check_cuda_reports_file_line_expression_and_message},
        {"cpu_timer", test_cpu_timer_measures_elapsed_milliseconds},
        {"run_benchmark", test_run_benchmark_records_warmups_trials_and_summary_stats},
        {"cuda_runtime", test_cuda_runtime_helpers_when_device_is_available},
    };

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
        return EXIT_FAILURE;
    }

    std::cout << tests.size() << " test(s) passed\n";
    return EXIT_SUCCESS;
}
