#include <coooda/core/cuda_checks.hpp>

#include <sstream>
#include <utility>

namespace coooda::core {
namespace {

std::string format_cuda_error(
    cudaError_t result,
    const char *expression,
    const char *file,
    int line
) {
    std::ostringstream message;
    message << "CUDA error at " << file << ':' << line << " while evaluating '" << expression
            << "': " << cudaGetErrorString(result) << " (code=" << static_cast<int>(result) << ')';
    return message.str();
}

} // namespace

CudaError::CudaError(
    cudaError_t code,
    std::string message
)
    : std::runtime_error(std::move(message)), code_(code) {}

cudaError_t CudaError::code() const noexcept { return code_; }

void check_cuda(
    cudaError_t result,
    const char *expression,
    const char *file,
    int line
) {
    if (result != cudaSuccess) {
        throw CudaError(result, format_cuda_error(result, expression, file, line));
    }
}

void check_last_kernel(
    const char *kernel_name,
    const char *file,
    int line
) {
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::ostringstream expression;
        expression << "kernel launch '" << kernel_name << "'";
        throw CudaError(result, format_cuda_error(result, expression.str().c_str(), file, line));
    }
}

void synchronize_device() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", "synchronize_device", 0);
}

void synchronize_device(
    const char *file,
    int line
) {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", file, line);
}

} // namespace coooda::core
