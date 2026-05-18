#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/memory/device_buffer.cuh>

#include <vector>

namespace {

std::vector<float> copy_tensor_values(const coooda_core::Tensor &tensor) {
    std::vector<float> values(tensor.size(), 0.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = tensor[i];
    }
    return values;
}

void fill_tensor(coooda_core::Tensor &tensor, const std::vector<float> &values) {
    if (tensor.size() != values.size()) {
        coooda_core::fail("memory_tensors benchmark tensor fill size mismatch");
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        tensor[i] = values[i];
    }
}

void run_memory_tensors_case() {
    const coooda_core::Shape shape{{128, 128}};
    coooda_core::Tensor host_tensor(shape);
    const std::vector<float> expected =
        coooda_core::test::seeded_vector(host_tensor.size(), 0xC0DAU, -1.0f, 1.0f);
    fill_tensor(host_tensor, expected);

    coooda_core::Tensor actual_tensor(shape);
    const auto result = coooda_core::bench::time_once("memory_tensors_round_trip", [&]() {
        coooda_cuda::memory::DeviceBuffer device_tensor(host_tensor.size());
        device_tensor.copy_from_host(host_tensor.data(), host_tensor.size());
        device_tensor.copy_to_host(actual_tensor.data(), actual_tensor.size());
    });

    const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
        "memory_tensors_round_trip",
        expected,
        copy_tensor_values(actual_tensor),
        0.0f,
        0.0f
    );
    if (!report.matched) {
        coooda_core::fail(report.to_string());
    }

    coooda_core::bench::print_result(result);
}

} // namespace

namespace coooda_bench::compare {

void append_memory_tensor_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "memory_tensors",
        "host-device tensor transfer benchmark",
        []() { run_memory_tensors_case(); },
    });
}

} // namespace coooda_bench::compare
