#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-5f;
constexpr float kRelTol = 1.0e-5f;

struct BinaryCase {
    std::string label;
    std::vector<float> a;
    std::vector<float> b;
};

struct SaxpyCase {
    std::string label;
    float alpha = 0.0f;
    std::vector<float> x;
    std::vector<float> y;
};

struct UnaryCase {
    std::string label;
    std::vector<float> input;
};

bool compare_outputs(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);

    if (!report.matched) {
        std::cerr << report.to_string() << "\n";
        return false;
    }
    return true;
}

bool vector_add_case_matches(const BinaryCase &test_case) {
    const std::vector<float> expected = coooda_cpp::ops::vector_add_reference(test_case.a, test_case.b);
    const std::vector<float> baseline = coooda_cuda::ops::vector_add_baseline(test_case.a, test_case.b);
    const std::vector<float> grid_stride = coooda_cuda::ops::vector_add_grid_stride(test_case.a, test_case.b);
    return compare_outputs(test_case.label + "_baseline", expected, baseline) &&
           compare_outputs(test_case.label + "_grid_stride", expected, grid_stride) &&
           compare_outputs(test_case.label + "_baseline_vs_grid_stride", baseline, grid_stride);
}

bool saxpy_case_matches(const SaxpyCase &test_case) {
    const std::vector<float> expected =
        coooda_cpp::ops::saxpy_reference(test_case.alpha, test_case.x, test_case.y);
    const std::vector<float> baseline =
        coooda_cuda::ops::saxpy_baseline(test_case.alpha, test_case.x, test_case.y);
    const std::vector<float> grid_stride =
        coooda_cuda::ops::saxpy_grid_stride(test_case.alpha, test_case.x, test_case.y);
    return compare_outputs(test_case.label + "_baseline", expected, baseline) &&
           compare_outputs(test_case.label + "_grid_stride", expected, grid_stride) &&
           compare_outputs(test_case.label + "_baseline_vs_grid_stride", baseline, grid_stride);
}

bool relu_case_matches(const UnaryCase &test_case) {
    const std::vector<float> expected = coooda_cpp::ops::relu_reference(test_case.input);
    const std::vector<float> baseline = coooda_cuda::ops::relu_baseline(test_case.input);
    const std::vector<float> grid_stride = coooda_cuda::ops::relu_grid_stride(test_case.input);
    return compare_outputs(test_case.label + "_baseline", expected, baseline) &&
           compare_outputs(test_case.label + "_grid_stride", expected, grid_stride) &&
           compare_outputs(test_case.label + "_baseline_vs_grid_stride", baseline, grid_stride);
}

bool sigmoid_case_matches(const UnaryCase &test_case) {
    const std::vector<float> expected = coooda_cpp::ops::sigmoid_reference(test_case.input);
    const std::vector<float> baseline = coooda_cuda::ops::sigmoid_baseline(test_case.input);
    const std::vector<float> grid_stride = coooda_cuda::ops::sigmoid_grid_stride(test_case.input);
    return compare_outputs(test_case.label + "_baseline", expected, baseline) &&
           compare_outputs(test_case.label + "_grid_stride", expected, grid_stride) &&
           compare_outputs(test_case.label + "_baseline_vs_grid_stride", baseline, grid_stride);
}

bool elementwise_multiply_case_matches(const BinaryCase &test_case) {
    const std::vector<float> expected =
        coooda_cpp::ops::elementwise_multiply_reference(test_case.a, test_case.b);
    const std::vector<float> baseline =
        coooda_cuda::ops::elementwise_multiply_baseline(test_case.a, test_case.b);
    const std::vector<float> grid_stride =
        coooda_cuda::ops::elementwise_multiply_grid_stride(test_case.a, test_case.b);
    return compare_outputs(test_case.label + "_baseline", expected, baseline) &&
           compare_outputs(test_case.label + "_grid_stride", expected, grid_stride) &&
           compare_outputs(test_case.label + "_baseline_vs_grid_stride", baseline, grid_stride);
}

} // namespace

namespace coooda_compare::ops {

bool vector_add_matches_reference() {
    const std::vector<BinaryCase> test_cases{
        {"vector_add_empty", {}, {}},
        {"vector_add_exact_small", {1.0f, -2.0f, 3.5f, 4.0f}, {0.5f, 2.0f, -1.5f, 8.0f}},
        {"vector_add_boundary", {0.0f, -0.0f, 1.0e-6f, -1000.0f}, {0.0f, 5.0f, -1.0e-6f, 1000.0f}},
        {"vector_add_seeded_32",
         coooda_core::test::seeded_vector(32, 0xC001U, -10.0f, 10.0f),
         coooda_core::test::seeded_vector(32, 0xD00DU, -10.0f, 10.0f)},
        {"vector_add_seeded_1024",
         coooda_core::test::seeded_vector(1024, 0xA11CEU, -100.0f, 100.0f),
         coooda_core::test::seeded_vector(1024, 0xB0BU, -100.0f, 100.0f)},
    };

    for (const BinaryCase &test_case : test_cases) {
        if (!vector_add_case_matches(test_case)) {
            return false;
        }
    }
    return true;
}

bool saxpy_matches_reference() {
    const std::vector<SaxpyCase> test_cases{
        {"saxpy_empty", 2.0f, {}, {}},
        {"saxpy_exact_small", 2.0f, {1.0f, -2.0f, 4.0f}, {3.0f, 1.0f, 1.0f}},
        {"saxpy_boundary", -0.5f, {0.0f, -0.0f, 1.0e-6f, -1000.0f}, {1.0f, -1.0f, 2.0f, 1000.0f}},
        {"saxpy_seeded_64",
         1.25f,
         coooda_core::test::seeded_vector(64, 0x5A11U, -10.0f, 10.0f),
         coooda_core::test::seeded_vector(64, 0x9A1U, -10.0f, 10.0f)},
        {"saxpy_seeded_1024",
         -0.75f,
         coooda_core::test::seeded_vector(1024, 0xA11CEU, -100.0f, 100.0f),
         coooda_core::test::seeded_vector(1024, 0xC0DAU, -100.0f, 100.0f)},
    };

    for (const SaxpyCase &test_case : test_cases) {
        if (!saxpy_case_matches(test_case)) {
            return false;
        }
    }
    return true;
}

bool relu_matches_reference() {
    const std::vector<UnaryCase> test_cases{
        {"relu_empty", {}},
        {"relu_exact_small", {-3.0f, 0.0f, 2.5f, 10.0f}},
        {"relu_boundary", {-1000.0f, -1.0e-6f, 0.0f, 1.0e-6f, 1000.0f}},
        {"relu_seeded_64", coooda_core::test::seeded_vector(64, 0x7E1U, -10.0f, 10.0f)},
        {"relu_seeded_1024", coooda_core::test::seeded_vector(1024, 0x7E1U, -100.0f, 100.0f)},
    };

    for (const UnaryCase &test_case : test_cases) {
        if (!relu_case_matches(test_case)) {
            return false;
        }
    }
    return true;
}

bool sigmoid_matches_reference() {
    const std::vector<UnaryCase> test_cases{
        {"sigmoid_empty", {}},
        {"sigmoid_exact_small", {-2.0f, 0.0f, 2.0f}},
        {"sigmoid_boundary", {-20.0f, -1.0e-6f, 0.0f, 1.0e-6f, 20.0f}},
        {"sigmoid_seeded_64", coooda_core::test::seeded_vector(64, 0x519U, -10.0f, 10.0f)},
        {"sigmoid_seeded_1024", coooda_core::test::seeded_vector(1024, 0x519U, -20.0f, 20.0f)},
    };

    for (const UnaryCase &test_case : test_cases) {
        if (!sigmoid_case_matches(test_case)) {
            return false;
        }
    }
    return true;
}

bool elementwise_multiply_matches_reference() {
    const std::vector<BinaryCase> test_cases{
        {"elementwise_multiply_empty", {}, {}},
        {"elementwise_multiply_exact_small", {1.0f, -2.0f, 0.0f}, {3.0f, 4.0f, -5.0f}},
        {"elementwise_multiply_boundary", {0.0f, -0.0f, 1.0e-6f, -1000.0f}, {5.0f, -5.0f, -1.0e-6f, 0.25f}},
        {"elementwise_multiply_seeded_64",
         coooda_core::test::seeded_vector(64, 0xDA7AU, -10.0f, 10.0f),
         coooda_core::test::seeded_vector(64, 0xBEEFU, -10.0f, 10.0f)},
        {"elementwise_multiply_seeded_1024",
         coooda_core::test::seeded_vector(1024, 0xDA7AU, -100.0f, 100.0f),
         coooda_core::test::seeded_vector(1024, 0xBEEFU, -100.0f, 100.0f)},
    };

    for (const BinaryCase &test_case : test_cases) {
        if (!elementwise_multiply_case_matches(test_case)) {
            return false;
        }
    }
    return true;
}

bool vector_ops_match_reference() {
    return vector_add_matches_reference() &&
           saxpy_matches_reference() &&
           relu_matches_reference() &&
           sigmoid_matches_reference() &&
           elementwise_multiply_matches_reference();
}

} // namespace coooda_compare::ops
