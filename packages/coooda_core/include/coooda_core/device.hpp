#pragma once

#include <cstddef>
#include <string>

namespace coooda_core::cuda {

struct DeviceInfo {
    int id;
    std::string name;
    std::size_t global_memory_bytes;
    int warp_size;
};

int device_count();
DeviceInfo get_device_info(int device_id);
std::string device_summary();

} // namespace coooda_core::cuda
