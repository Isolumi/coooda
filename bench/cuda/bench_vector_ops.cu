#include <coooda_core/benchmark.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <vector>

namespace {

struct VectorInputs {
    std::vector<float> a;
    std::vector<float> b;
};

VectorInputs make_inputs(std::size_t size) {
    return VectorInputs{
        coooda_core::test::seeded_vector(size, 0xC001U, -10.0f, 10.0f),
        coooda_core::test::seeded_vector(size, 0xD00DU, -10.0f, 10.0f),
    };
}

void run_baseline_ops(const VectorInputs &inputs) {
    (void)coooda_cuda::ops::vector_add_baseline(inputs.a, inputs.b);
    (void)coooda_cuda::ops::saxpy_baseline(1.25f, inputs.a, inputs.b);
    (void)coooda_cuda::ops::relu_baseline(inputs.a);
    (void)coooda_cuda::ops::sigmoid_baseline(inputs.a);
    (void)coooda_cuda::ops::elementwise_multiply_baseline(inputs.a, inputs.b);
}

void run_grid_stride_ops(const VectorInputs &inputs) {
    (void)coooda_cuda::ops::vector_add_grid_stride(inputs.a, inputs.b);
    (void)coooda_cuda::ops::saxpy_grid_stride(1.25f, inputs.a, inputs.b);
    (void)coooda_cuda::ops::relu_grid_stride(inputs.a);
    (void)coooda_cuda::ops::sigmoid_grid_stride(inputs.a);
    (void)coooda_cuda::ops::elementwise_multiply_grid_stride(inputs.a, inputs.b);
}

void run_vector_ops_case() {
    const VectorInputs inputs = make_inputs(1 << 18);

    coooda_core::bench::print_result(coooda_core::bench::time_once("cuda_vector_ops_baseline", [&]() {
        run_baseline_ops(inputs);
    }));

    coooda_core::bench::print_result(coooda_core::bench::time_once("cuda_vector_ops_grid_stride", [&]() {
        run_grid_stride_ops(inputs);
    }));
}

} // namespace

namespace coooda_bench::cuda {

void append_vector_ops_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "vector_ops",
        "CUDA baseline and grid-stride vector ops benchmark",
        []() { run_vector_ops_case(); },
    });
}

} // namespace coooda_bench::cuda
