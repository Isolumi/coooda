#include <coooda_core/cuda_check.cuh>

#include <sstream>

namespace coooda_core::cuda {

CudaError::CudaError(cudaError_t code, const std::string &message)
    : std::runtime_error(message), code_(code) {}

cudaError_t CudaError::code() const {
    return code_;
}

void check(cudaError_t code, const char *expr, const char *file, int line) {
    if (code == cudaSuccess) {
        return;
    }

    std::ostringstream out;
    out << file << ":" << line << " CUDA call failed: " << expr << ": " << cudaGetErrorString(code);
    throw CudaError(code, out.str());
}

void check_last_kernel(const char *kernel, const char *file, int line) {
    check(cudaGetLastError(), kernel, file, line);
}

} // namespace coooda_core::cuda
