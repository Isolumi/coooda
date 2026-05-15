#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace coooda_core::cuda {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string &message);
    cudaError_t code() const;

private:
    cudaError_t code_;
};

void check(cudaError_t code, const char *expr, const char *file, int line);
void check_last_kernel(const char *kernel, const char *file, int line);

} // namespace coooda_core::cuda

#define COODA_CUDA_CHECK(expr) ::coooda_core::cuda::check((expr), #expr, __FILE__, __LINE__)
#define COODA_CUDA_CHECK_LAST(kernel) ::coooda_core::cuda::check_last_kernel((kernel), __FILE__, __LINE__)
