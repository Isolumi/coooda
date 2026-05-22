#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/elementwise.hpp>
#include <coooda_cuda/ops/elementwise.cuh>

#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-4f;
constexpr float kRelTol = 1.0e-5f;

struct ElementwiseInputs {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> bias;
};

ElementwiseInputs make_inputs() {
    const std::size_t size = 64 * 1024;
    const std::size_t bias_size = 256;
    return ElementwiseInputs{
        coooda_core::test::seeded_vector(size, 0xA11CEU, -3.0f, 3.0f),
        coooda_core::test::seeded_vector(size, 0xB0BU, -3.0f, 3.0f),
        coooda_core::test::seeded_vector(bias_size, 0xC001U, -1.0f, 1.0f),
    };
}

void require_vector_close(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    if (!report.matched) {
        coooda_core::fail(report.to_string());
    }
}

void compare_baseline_ops(const ElementwiseInputs &inputs) {
    require_vector_close(
        "elementwise_baseline_relu",
        coooda_cpp::ops::unary_relu_reference(inputs.a),
        coooda_cuda::ops::unary_relu_baseline(inputs.a)
    );
    require_vector_close(
        "elementwise_baseline_gelu",
        coooda_cpp::ops::unary_gelu_reference(inputs.a),
        coooda_cuda::ops::unary_gelu_baseline(inputs.a)
    );
    require_vector_close(
        "elementwise_baseline_add",
        coooda_cpp::ops::binary_add_reference(inputs.a, inputs.b),
        coooda_cuda::ops::binary_add_baseline(inputs.a, inputs.b)
    );
    require_vector_close(
        "elementwise_baseline_multiply",
        coooda_cpp::ops::binary_multiply_reference(inputs.a, inputs.b),
        coooda_cuda::ops::binary_multiply_baseline(inputs.a, inputs.b)
    );
    require_vector_close(
        "elementwise_baseline_scalar",
        coooda_cpp::ops::add_scalar_reference(inputs.a, 1.25f),
        coooda_cuda::ops::add_scalar_baseline(inputs.a, 1.25f)
    );
    require_vector_close(
        "elementwise_baseline_bias",
        coooda_cpp::ops::add_bias_reference(inputs.a, inputs.bias),
        coooda_cuda::ops::add_bias_baseline(inputs.a, inputs.bias)
    );
}

void compare_grid_stride_ops(const ElementwiseInputs &inputs) {
    require_vector_close(
        "elementwise_grid_stride_relu",
        coooda_cpp::ops::unary_relu_reference(inputs.a),
        coooda_cuda::ops::unary_relu_grid_stride(inputs.a)
    );
    require_vector_close(
        "elementwise_grid_stride_gelu",
        coooda_cpp::ops::unary_gelu_reference(inputs.a),
        coooda_cuda::ops::unary_gelu_grid_stride(inputs.a)
    );
    require_vector_close(
        "elementwise_grid_stride_add",
        coooda_cpp::ops::binary_add_reference(inputs.a, inputs.b),
        coooda_cuda::ops::binary_add_grid_stride(inputs.a, inputs.b)
    );
    require_vector_close(
        "elementwise_grid_stride_multiply",
        coooda_cpp::ops::binary_multiply_reference(inputs.a, inputs.b),
        coooda_cuda::ops::binary_multiply_grid_stride(inputs.a, inputs.b)
    );
    require_vector_close(
        "elementwise_grid_stride_scalar",
        coooda_cpp::ops::add_scalar_reference(inputs.a, 1.25f),
        coooda_cuda::ops::add_scalar_grid_stride(inputs.a, 1.25f)
    );
    require_vector_close(
        "elementwise_grid_stride_bias",
        coooda_cpp::ops::add_bias_reference(inputs.a, inputs.bias),
        coooda_cuda::ops::add_bias_grid_stride(inputs.a, inputs.bias)
    );
}

void compare_unfused_bias_relu(const ElementwiseInputs &inputs) {
    const std::vector<float> bias_added = coooda_cuda::ops::add_bias_grid_stride(inputs.a, inputs.bias);
    require_vector_close(
        "elementwise_unfused_bias_relu",
        coooda_cpp::ops::add_bias_relu_reference(inputs.a, inputs.bias),
        coooda_cuda::ops::unary_relu_grid_stride(bias_added)
    );
}

void compare_unfused_bias_gelu(const ElementwiseInputs &inputs) {
    const std::vector<float> bias_added = coooda_cuda::ops::add_bias_grid_stride(inputs.a, inputs.bias);
    require_vector_close(
        "elementwise_unfused_bias_gelu",
        coooda_cpp::ops::add_bias_gelu_reference(inputs.a, inputs.bias),
        coooda_cuda::ops::unary_gelu_grid_stride(bias_added)
    );
}

void compare_fused_bias_relu(const ElementwiseInputs &inputs) {
    require_vector_close(
        "elementwise_fused_bias_relu",
        coooda_cpp::ops::add_bias_relu_reference(inputs.a, inputs.bias),
        coooda_cuda::ops::add_bias_relu_grid_stride(inputs.a, inputs.bias)
    );
}

void compare_fused_bias_gelu(const ElementwiseInputs &inputs) {
    require_vector_close(
        "elementwise_fused_bias_gelu",
        coooda_cpp::ops::add_bias_gelu_reference(inputs.a, inputs.bias),
        coooda_cuda::ops::add_bias_gelu_grid_stride(inputs.a, inputs.bias)
    );
}

void warm_up_cuda() {
    const std::vector<float> a{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> b{0.25f, 2.0f, -1.5f, 0.5f};
    const std::vector<float> bias{0.5f, -1.0f};

    (void)coooda_cuda::ops::unary_relu_baseline(a);
    (void)coooda_cuda::ops::unary_relu_grid_stride(a);
    (void)coooda_cuda::ops::binary_add_baseline(a, b);
    (void)coooda_cuda::ops::binary_add_grid_stride(a, b);
    (void)coooda_cuda::ops::add_bias_relu_baseline(a, bias);
    (void)coooda_cuda::ops::add_bias_relu_grid_stride(a, bias);
}

void run_elementwise_compare_case() {
    const ElementwiseInputs inputs = make_inputs();
    warm_up_cuda();

    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_elementwise_baseline", [&]() {
        compare_baseline_ops(inputs);
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_elementwise_grid_stride", [&]() {
        compare_grid_stride_ops(inputs);
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_elementwise_unfused_bias_relu", [&]() {
        compare_unfused_bias_relu(inputs);
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_elementwise_fused_bias_relu", [&]() {
        compare_fused_bias_relu(inputs);
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_elementwise_unfused_bias_gelu", [&]() {
        compare_unfused_bias_gelu(inputs);
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_elementwise_fused_bias_gelu", [&]() {
        compare_fused_bias_gelu(inputs);
    }));
}

} // namespace

namespace coooda_bench::compare {

void append_elementwise_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "elementwise",
        "compare C++ reference, CUDA baseline, CUDA grid-stride, and fused elementwise ops",
        []() { run_elementwise_compare_case(); },
    });
}

} // namespace coooda_bench::compare
