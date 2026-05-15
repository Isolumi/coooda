#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace coooda::core {

class CudaError : public std::runtime_error {
  public:
    CudaError(cudaError_t code, std::string message);

    cudaError_t code() const noexcept;

  private:
    cudaError_t code_;
};

void check_cuda(cudaError_t result, const char *expression, const char *file, int line);

void check_last_kernel(const char *kernel_name, const char *file, int line);

void synchronize_device();
void synchronize_device(const char *file, int line);

} // namespace coooda::core

#define COODA_CHECK_CUDA(call) ::coooda::core::check_cuda((call), #call, __FILE__, __LINE__)

#define COODA_CHECK_LAST_KERNEL(kernel_name)                                                       \
    ::coooda::core::check_last_kernel((kernel_name), __FILE__, __LINE__)

#define COODA_SYNCHRONIZE_DEVICE() ::coooda::core::synchronize_device(__FILE__, __LINE__)
