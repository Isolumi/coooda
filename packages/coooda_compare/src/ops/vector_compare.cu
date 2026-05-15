#include <coooda_compare/ops/vector_compare.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/vector.hpp>
#include <coooda_cuda/ops/vector.cuh>

#include <iostream>
#include <string>
#include <vector>

namespace {

struct VectorAddCase {
    std::string label;
    std::vector<float> a;
    std::vector<float> b;
};

bool vector_add_case_matches(const VectorAddCase &test_case) {
    const std::vector<float> expected = coooda_cpp::ops::vector_add_reference(test_case.a, test_case.b);
    const std::vector<float> actual = coooda_cuda::ops::vector_add_baseline(test_case.a, test_case.b);
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(test_case.label, expected, actual, 1.0e-6f, 1.0e-6f);

    if (!report.matched) {
        std::cerr << report.to_string() << "\n";
        return false;
    }
    return true;
}

} // namespace

namespace coooda_compare::ops {

bool vector_add_matches_reference() {
    const std::vector<VectorAddCase> test_cases{
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

    for (const VectorAddCase &test_case : test_cases) {
        if (!vector_add_case_matches(test_case)) {
            return false;
        }
    }
    return true;
}

} // namespace coooda_compare::ops
