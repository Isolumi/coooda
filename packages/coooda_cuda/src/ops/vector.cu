#include <coooda_cuda/ops/vector.cuh>
#include <coooda_core/cuda_check.cuh>
#include <coooda_core/status.hpp>

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>

namespace {

class DeviceFloatBuffer {
public:
    DeviceFloatBuffer() = default;
    DeviceFloatBuffer(const DeviceFloatBuffer &) = delete;
    DeviceFloatBuffer &operator=(const DeviceFloatBuffer &) = delete;

    ~DeviceFloatBuffer() noexcept {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }

    void allocate(std::size_t bytes) {
        COODA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), bytes));
    }

    float *get() const noexcept {
        return ptr_;
    }

private:
    float *ptr_ = nullptr;
};

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

    if (a.size() > static_cast<std::size_t>(INT_MAX)) {
        coooda_core::fail("vector_add_baseline input is too large for the baseline kernel");
    }

    const std::size_t bytes = a.size() * sizeof(float);
    DeviceFloatBuffer device_a;
    DeviceFloatBuffer device_b;
    DeviceFloatBuffer device_out;

    device_a.allocate(bytes);
    device_b.allocate(bytes);
    device_out.allocate(bytes);
    COODA_CUDA_CHECK(cudaMemcpy(device_a.get(), a.data(), bytes, cudaMemcpyHostToDevice));
    COODA_CUDA_CHECK(cudaMemcpy(device_b.get(), b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int n = static_cast<int>(a.size());
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(device_a.get(), device_b.get(), device_out.get(), n);
    COODA_CUDA_CHECK_LAST("vector_add_kernel");

    std::vector<float> out(a.size(), 0.0f);
    COODA_CUDA_CHECK(cudaMemcpy(out.data(), device_out.get(), bytes, cudaMemcpyDeviceToHost));
    return out;
}

} // namespace coooda_cuda::ops
