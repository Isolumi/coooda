#include <coooda/bench/benchmark.hpp>
#include <coooda/core/cuda_checks.hpp>
#include <coooda/core/device.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#ifndef COODA_BUILD_MODE
#define COODA_BUILD_MODE "unknown"
#endif

namespace {

__global__ void noop_kernel() {}

coooda::bench::BenchmarkConfig base_config(
    const std::string &operation,
    const std::string &variant,
    const std::string &shape,
    const std::string &dtype,
    const std::string &gpu
) {
    coooda::bench::BenchmarkConfig config;
    config.operation = operation;
    config.variant = variant;
    config.shape = shape;
    config.dtype = dtype;
    config.build_mode = COODA_BUILD_MODE;
    config.gpu = gpu;
    config.warmups = 5;
    config.trials = 25;
    return config;
}

} // namespace

int main() {
    try {
        const int count = coooda::core::device_count();
        if (count == 0) {
            std::cout << "No CUDA devices found; benchmark skipped.\n";
            return 0;
        }

        COODA_CHECK_CUDA(cudaSetDevice(0));
        const coooda::core::DeviceInfo info = coooda::core::get_device_info(0);
        std::cout << info << "\n\n";

        const std::string gpu = info.name;

        coooda::bench::BenchmarkConfig noop =
            base_config("noop_launch", "single_empty_kernel", "<<<1,1>>>", "none", gpu);
        const coooda::bench::TimingStats noop_stats = coooda::bench::run_benchmark(noop, []() {
            noop_kernel<<<1, 1>>>();
            COODA_CHECK_LAST_KERNEL("noop_kernel");
        });
        std::cout << noop_stats << '\n';

        constexpr std::size_t copy_bytes = 16 * 1024 * 1024;
        std::vector<unsigned char> host(copy_bytes, 7);
        unsigned char *device = nullptr;
        COODA_CHECK_CUDA(cudaMalloc(&device, copy_bytes));

        coooda::bench::BenchmarkConfig copy =
            base_config("host_to_device_copy", "cudaMemcpy", "16MiB", "u8", gpu);
        const coooda::bench::TimingStats copy_stats =
            coooda::bench::run_benchmark(copy, [&host, device]() {
                COODA_CHECK_CUDA(
                    cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice)
                );
            });
        std::cout << copy_stats << '\n';

        COODA_CHECK_CUDA(cudaFree(device));

        coooda::bench::BenchmarkConfig sync =
            base_config("device_sync", "cudaDeviceSynchronize", "device", "none", gpu);
        sync.synchronize_before_each_trial = false;
        sync.synchronize_after_each_trial = false;
        const coooda::bench::TimingStats sync_stats =
            coooda::bench::run_benchmark(sync, []() { COODA_CHECK_CUDA(cudaDeviceSynchronize()); });
        std::cout << sync_stats << '\n';

        return 0;
    } catch (const coooda::core::CudaError &error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
