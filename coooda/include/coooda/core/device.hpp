#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

namespace coooda::core {

struct DeviceInfo {
    int id = -1;
    std::string name;
    int major = 0;
    int minor = 0;
    std::size_t global_memory_bytes = 0;
    int multiprocessor_count = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    int clock_rate_khz = 0;
    int memory_clock_rate_khz = 0;
    int memory_bus_width_bits = 0;
    bool unified_addressing = false;

    std::string architecture() const;
};

int device_count();
DeviceInfo get_device_info(int device_id);

std::ostream &operator<<(std::ostream &os, const DeviceInfo &info);

} // namespace coooda::core
