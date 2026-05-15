#include <coooda_core/device.hpp>
#include <coooda_core/cuda_check.cuh>

#include <cuda_runtime.h>

#include <sstream>

namespace coooda_core::cuda {

int device_count() {
    int count = 0;
    COODA_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp prop{};
    COODA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return DeviceInfo{device_id, prop.name, prop.totalGlobalMem, prop.warpSize};
}

std::string device_summary() {
    const int count = device_count();
    if (count == 0) {
        return "No CUDA devices found.";
    }

    const DeviceInfo info = get_device_info(0);
    std::ostringstream out;
    out << "CUDA device 0: " << info.name
        << ", global memory bytes=" << info.global_memory_bytes
        << ", warp size=" << info.warp_size;
    return out.str();
}

} // namespace coooda_core::cuda
