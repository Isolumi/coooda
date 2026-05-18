#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <vector>

int main() {
    return coooda_core::test::run_tests({
        {"device_buffer_default_is_empty", []() {
             const coooda_cuda::memory::DeviceBuffer buffer;

             coooda_core::test::require(buffer.empty(), "default device buffer should be empty");
             coooda_core::test::require(buffer.size() == 0, "default device buffer size");
             coooda_core::test::require(buffer.bytes() == 0, "default device buffer bytes");
             coooda_core::test::require(buffer.data() == nullptr, "default device buffer data pointer");
         }},

        {"device_buffer_allocates_storage", []() {
             const coooda_cuda::memory::DeviceBuffer buffer(4);

             coooda_core::test::require(!buffer.empty(), "allocated device buffer should not be empty");
             coooda_core::test::require(buffer.size() == 4, "allocated device buffer size");
             coooda_core::test::require(buffer.bytes() == 4 * sizeof(float), "allocated device buffer bytes");
             coooda_core::test::require(buffer.data() != nullptr, "allocated device buffer data pointer");
         }},

        {"device_buffer_copies_from_host_to_device_and_back", []() {
             const std::vector<float> expected{1.0f, 2.0f, 3.5f, -4.0f};
             coooda_cuda::memory::DeviceBuffer buffer(expected.size());

             buffer.copy_from_host(expected.data(), expected.size());

             std::vector<float> actual(expected.size(), 0.0f);
             buffer.copy_to_host(actual.data(), actual.size());

             for (std::size_t i = 0; i < expected.size(); ++i) {
                 coooda_core::test::require(actual[i] == expected[i], "device buffer round trip value");
             }
         }},

        {"device_buffer_builds_from_host_vector", []() {
             const std::vector<float> expected{8.0f, -2.0f, 0.5f};

             coooda_cuda::memory::DeviceBuffer buffer =
                 coooda_cuda::memory::DeviceBuffer::from_host(expected);
             const std::vector<float> actual = buffer.copy_to_host();

             coooda_core::test::require(actual.size() == expected.size(), "copied host vector size");
             for (std::size_t i = 0; i < expected.size(); ++i) {
                 coooda_core::test::require(actual[i] == expected[i], "copied host vector value");
             }
         }},

        {"device_buffer_empty_copy_is_noop", []() {
             coooda_cuda::memory::DeviceBuffer buffer;
             const std::vector<float> actual = buffer.copy_to_host();

             coooda_core::test::require(actual.empty(), "empty device buffer copy should be empty");
             buffer.copy_from_host(nullptr, 0);
             buffer.copy_to_host(nullptr, 0);
         }},
    });
}
