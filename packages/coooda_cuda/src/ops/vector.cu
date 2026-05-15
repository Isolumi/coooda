#include <coooda_cuda/ops/vector.cuh>
#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>

#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void vector_add_kernel(const float *a, const float *b, float *out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

} // namespace

namespace coooda_cuda::ops {

std::vector<float> vector_add_baseline(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        coooda_core::fail("vector_add_baseline requires equal input sizes");
    }

    if (a.empty()) {
        return {};
    }

    const std::size_t bytes = a.size() * sizeof(float);
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_out = nullptr;

    COODA_CUDA_CHECK(cudaMalloc(&device_a, bytes));
    COODA_CUDA_CHECK(cudaMalloc(&device_b, bytes));
    COODA_CUDA_CHECK(cudaMalloc(&device_out, bytes));
    COODA_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    COODA_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int n = static_cast<int>(a.size());
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(device_a, device_b, device_out, n);
    COODA_CUDA_CHECK_LAST("vector_add_kernel");

    std::vector<float> out(a.size(), 0.0f);
    COODA_CUDA_CHECK(cudaMemcpy(out.data(), device_out, bytes, cudaMemcpyDeviceToHost));
    COODA_CUDA_CHECK(cudaFree(device_a));
    COODA_CUDA_CHECK(cudaFree(device_b));
    COODA_CUDA_CHECK(cudaFree(device_out));
    return out;
}

} // namespace coooda_cuda::ops
