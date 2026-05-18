#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <vector>

namespace {

void require_matched(const coooda_core::test::MismatchReport &report) {
    coooda_core::test::require(report.matched, report.to_string());
}

void fill_tensor(coooda_core::Tensor &tensor, const std::vector<float> &values) {
    coooda_core::test::require(tensor.size() == values.size(), "tensor fill size");
    for (std::size_t i = 0; i < values.size(); ++i) {
        tensor[i] = values[i];
    }
}

std::vector<float> copy_tensor_values(const coooda_core::Tensor &tensor) {
    std::vector<float> values(tensor.size(), 0.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = tensor[i];
    }
    return values;
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"memory_tensors_host_to_device_round_trip", []() {
             coooda_core::Tensor host_tensor({{4, 8}});
             const std::vector<float> expected =
                 coooda_core::test::seeded_vector(host_tensor.size(), 0xA11CEU, -10.0f, 10.0f);
             fill_tensor(host_tensor, expected);

             coooda_cuda::memory::DeviceBuffer device_tensor(host_tensor.size());
             device_tensor.copy_from_host(host_tensor.data(), host_tensor.size());
             const std::vector<float> actual = device_tensor.copy_to_host();

             require_matched(coooda_core::test::compare_vectors(
                 "memory_tensors_host_to_device_round_trip",
                 expected,
                 actual,
                 0.0f,
                 0.0f
             ));
         }},

        {"memory_tensors_device_to_host_tensor_round_trip", []() {
             const coooda_core::Shape shape{{2, 3, 4}};
             const std::vector<float> expected =
                 coooda_core::test::seeded_vector(coooda_core::numel(shape), 0xB0BU, -1.0f, 1.0f);

             coooda_cuda::memory::DeviceBuffer device_tensor =
                 coooda_cuda::memory::DeviceBuffer::from_host(expected);
             coooda_core::Tensor actual_tensor(shape);
             device_tensor.copy_to_host(actual_tensor.data(), actual_tensor.size());

             require_matched(coooda_core::test::compare_vectors(
                 "memory_tensors_device_to_host_tensor_round_trip",
                 expected,
                 copy_tensor_values(actual_tensor),
                 0.0f,
                 0.0f
             ));
         }},

        {"memory_tensors_tensor_view_round_trip", []() {
             coooda_core::Tensor host_tensor({{5}});
             const std::vector<float> expected =
                 coooda_core::test::seeded_vector(host_tensor.size(), 0x5EEDU, -5.0f, 5.0f);
             fill_tensor(host_tensor, expected);

             coooda_core::TensorView view = host_tensor.view();
             coooda_cuda::memory::DeviceBuffer device_tensor(view.size());
             device_tensor.copy_from_host(view.data(), view.size());

             std::vector<float> actual(view.size(), 0.0f);
             device_tensor.copy_to_host(actual.data(), actual.size());

             require_matched(coooda_core::test::compare_vectors(
                 "memory_tensors_tensor_view_round_trip",
                 expected,
                 actual,
                 0.0f,
                 0.0f
             ));
         }},

        {"memory_tensors_empty_round_trip", []() {
             coooda_core::Tensor host_tensor;
             coooda_cuda::memory::DeviceBuffer device_tensor(host_tensor.size());
             const std::vector<float> actual = device_tensor.copy_to_host();

             coooda_core::test::require(host_tensor.empty(), "host tensor should be empty");
             coooda_core::test::require(device_tensor.empty(), "device tensor should be empty");
             require_matched(coooda_core::test::compare_vectors(
                 "memory_tensors_empty_round_trip",
                 {},
                 actual,
                 0.0f,
                 0.0f
             ));
         }},
    });
}
