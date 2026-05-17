#include <coooda_core/cuda_check.cuh>
#include <coooda_core/device.hpp>
#include <coooda_core/test_harness.hpp>

#include <cuda_runtime.h>

#include <string>

namespace {

bool contains(
    const std::string &text,
    const std::string &needle
) {
    return text.find(needle) != std::string::npos;
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"cuda_check_accepts_success", []() { COODA_CUDA_CHECK(cudaSuccess); }},

        {"cuda_check_throws_cuda_error",
         []() {
             bool threw = false;

             try {
                 coooda_core::cuda::check(
                     cudaErrorInvalidValue, "fake_cuda_call()", "fake_file.cu", 123
                 );
             } catch (const coooda_core::cuda::CudaError &error) {
                 threw = true;
                 const std::string message = error.what();

                 coooda_core::test::require(
                     error.code() == cudaErrorInvalidValue,
                     "CudaError should store the original CUDA error code"
                 );
                 coooda_core::test::require(
                     contains(message, "fake_file.cu:123"),
                     "CudaError message should include file and line"
                 );
                 coooda_core::test::require(
                     contains(message, "fake_cuda_call()"),
                     "CudaError message should include the failed expression"
                 );
                 coooda_core::test::require(
                     contains(message, cudaGetErrorString(cudaErrorInvalidValue)),
                     "CudaError message should include CUDA runtime error text"
                 );
             }

             coooda_core::test::require(threw, "check should throw CudaError for CUDA failures");
         }},

        {"cuda_check_last_kernel_accepts_clean_state",
         []() {
             (void)cudaGetLastError();
             COODA_CUDA_CHECK_LAST("no_kernel_launched");
         }},

        {"cuda_device_helpers_report_or_find_device", []() {
             try {
                 const int count = coooda_core::cuda::device_count();
                 coooda_core::test::require(count >= 0, "device count should not be negative");

                 const std::string summary = coooda_core::cuda::device_summary();
                 coooda_core::test::require(!summary.empty(), "device summary should not be empty");

                 if (count > 0) {
                     const coooda_core::cuda::DeviceInfo info =
                         coooda_core::cuda::get_device_info(0);
                     coooda_core::test::require(
                         info.id == 0, "device id should match requested id"
                     );
                     coooda_core::test::require(
                         !info.name.empty(), "device name should not be empty"
                     );
                     coooda_core::test::require(
                         info.global_memory_bytes > 0, "device global memory should be positive"
                     );
                     coooda_core::test::require(
                         info.warp_size > 0, "device warp size should be positive"
                     );
                     coooda_core::test::require(
                         contains(summary, "CUDA device 0:"),
                         "device summary should include device 0"
                     );
                 }
             } catch (const coooda_core::cuda::CudaError &error) {
                 const std::string message = error.what();
                 coooda_core::test::require(
                     error.code() != cudaSuccess,
                     "device helper failure should preserve a non-success CUDA code"
                 );
                 coooda_core::test::require(
                     contains(message, "CUDA call failed"),
                     "device helper failure should report a clean CUDA error"
                 );
             }
         }},
    });
}
