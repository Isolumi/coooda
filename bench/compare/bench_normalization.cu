#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/normalization.hpp>
#include <coooda_cuda/nn/normalization.cuh>

#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-3f;
constexpr float kRelTol = 1.0e-5f;
constexpr float kEpsilon = 1.0e-5f;

struct NormalizationInputs {
    std::vector<float> input;
    std::vector<float> gamma;
    std::vector<float> beta;
    std::size_t feature_count = 0;
};

struct NormalizationExpected {
    std::vector<float> layer;
    std::vector<float> rms;
};

NormalizationInputs make_inputs() {
    constexpr std::size_t row_count = 256;
    constexpr std::size_t feature_count = 2048;

    return NormalizationInputs{
        coooda_core::test::seeded_vector(row_count * feature_count, 0x901U, -4.0f, 4.0f),
        coooda_core::test::seeded_vector(feature_count, 0x902U, 0.5f, 1.5f),
        coooda_core::test::seeded_vector(feature_count, 0x903U, -0.25f, 0.25f),
        feature_count,
    };
}

NormalizationExpected make_expected(const NormalizationInputs &inputs) {
    return NormalizationExpected{
        coooda_cpp::nn::layer_norm_reference(
            inputs.input,
            inputs.gamma,
            inputs.beta,
            inputs.feature_count,
            kEpsilon
        ),
        coooda_cpp::nn::rms_norm_reference(
            inputs.input,
            inputs.gamma,
            inputs.feature_count,
            kEpsilon
        ),
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

void warm_up_cuda(const NormalizationInputs &inputs) {
    (void)coooda_cuda::nn::layer_norm_baseline(
        inputs.input,
        inputs.gamma,
        inputs.beta,
        inputs.feature_count,
        kEpsilon
    );
    (void)coooda_cuda::nn::layer_norm_welford(
        inputs.input,
        inputs.gamma,
        inputs.beta,
        inputs.feature_count,
        kEpsilon
    );
    (void)coooda_cuda::nn::rms_norm_baseline(
        inputs.input,
        inputs.gamma,
        inputs.feature_count,
        kEpsilon
    );
}

void run_normalization_compare_case() {
    const NormalizationInputs inputs = make_inputs();
    const NormalizationExpected expected = make_expected(inputs);
    warm_up_cuda(inputs);

    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_normalization_layer_norm_baseline", [&]() {
        require_vector_close(
            "compare_normalization_layer_norm_baseline",
            expected.layer,
            coooda_cuda::nn::layer_norm_baseline(
                inputs.input,
                inputs.gamma,
                inputs.beta,
                inputs.feature_count,
                kEpsilon
            )
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_normalization_layer_norm_welford", [&]() {
        require_vector_close(
            "compare_normalization_layer_norm_welford",
            expected.layer,
            coooda_cuda::nn::layer_norm_welford(
                inputs.input,
                inputs.gamma,
                inputs.beta,
                inputs.feature_count,
                kEpsilon
            )
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_normalization_rms_norm_baseline", [&]() {
        require_vector_close(
            "compare_normalization_rms_norm_baseline",
            expected.rms,
            coooda_cuda::nn::rms_norm_baseline(
                inputs.input,
                inputs.gamma,
                inputs.feature_count,
                kEpsilon
            )
        );
    }));
}

} // namespace

namespace coooda_bench::compare {

void append_normalization_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "normalization",
        "compare C++ reference, CUDA baseline, and CUDA Welford normalization paths",
        []() { run_normalization_compare_case(); },
    });
}

} // namespace coooda_bench::compare
