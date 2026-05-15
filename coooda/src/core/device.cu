#include <coooda/core/device.hpp>

#include <coooda/core/cuda_checks.hpp>

#include <ostream>
#include <sstream>
#include <stdexcept>

namespace coooda::core {
namespace {

int get_device_attribute(
    int device_id,
    cudaDeviceAttr attribute
) {
    int value = 0;
    COODA_CHECK_CUDA(cudaDeviceGetAttribute(&value, attribute, device_id));
    return value;
}

} // namespace

std::string DeviceInfo::architecture() const {
    std::ostringstream os;
    os << "sm_" << major << minor;
    return os.str();
}

int device_count() {
    int count = 0;
    COODA_CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo get_device_info(
    int device_id
) {
    const int count = device_count();
    if (device_id < 0 || device_id >= count) {
        std::ostringstream message;
        message << "CUDA device id " << device_id << " is outside [0, " << count << ')';
        throw std::out_of_range(message.str());
    }

    cudaDeviceProp properties{};
    COODA_CHECK_CUDA(cudaGetDeviceProperties(&properties, device_id));

    DeviceInfo info;
    info.id = device_id;
    info.name = properties.name;
    info.major = properties.major;
    info.minor = properties.minor;
    info.global_memory_bytes = properties.totalGlobalMem;
    info.multiprocessor_count = properties.multiProcessorCount;
    info.max_threads_per_block = properties.maxThreadsPerBlock;
    info.warp_size = properties.warpSize;
    info.clock_rate_khz = get_device_attribute(device_id, cudaDevAttrClockRate);
    info.memory_clock_rate_khz = get_device_attribute(device_id, cudaDevAttrMemoryClockRate);
    info.memory_bus_width_bits = properties.memoryBusWidth;
    info.unified_addressing = properties.unifiedAddressing != 0;
    return info;
}

std::ostream &operator<<(
    std::ostream &os,
    const DeviceInfo &info
) {
    os << "CUDA device " << info.id << ": " << info.name << '\n'
       << "  architecture: " << info.architecture() << '\n'
       << "  global memory: " << info.global_memory_bytes << " bytes\n"
       << "  multiprocessors: " << info.multiprocessor_count << '\n'
       << "  max threads per block: " << info.max_threads_per_block << '\n'
       << "  warp size: " << info.warp_size << '\n'
       << "  core clock: " << info.clock_rate_khz << " kHz\n"
       << "  memory clock: " << info.memory_clock_rate_khz << " kHz\n"
       << "  memory bus width: " << info.memory_bus_width_bits << " bits\n"
       << "  unified addressing: " << (info.unified_addressing ? "yes" : "no");
    return os;
}

} // namespace coooda::core
