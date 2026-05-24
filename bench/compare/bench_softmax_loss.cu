#include <coooda_core/benchmark.hpp>
#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/loss.hpp>
#include <coooda_cuda/nn/loss.cuh>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-3f;
constexpr float kRelTol = 1.0e-5f;

struct SoftmaxLossInputs {
    std::vector<float> single_logits;
    std::vector<float> batch_logits;
    std::vector<std::size_t> targets;
    std::size_t class_count = 0;
};

struct SoftmaxLossExpected {
    std::vector<float> softmax;
    std::vector<float> log_softmax;
    float cross_entropy = 0.0f;
    float mean_cross_entropy = 0.0f;
};

SoftmaxLossInputs make_inputs() {
    constexpr std::size_t single_size = 4096;
    constexpr std::size_t batch_size = 4096;
    constexpr std::size_t class_count = 128;

    std::vector<std::size_t> targets(batch_size, 0);
    for (std::size_t row = 0; row < targets.size(); ++row) {
        targets[row] = (row * 17) % class_count;
    }

    return SoftmaxLossInputs{
        coooda_core::test::seeded_vector(single_size, 0x5017U, -8.0f, 8.0f),
        coooda_core::test::seeded_vector(batch_size * class_count, 0xA11CEU, -8.0f, 8.0f),
        targets,
        class_count,
    };
}

SoftmaxLossExpected make_expected(const SoftmaxLossInputs &inputs) {
    return SoftmaxLossExpected{
        coooda_cpp::nn::softmax_reference(inputs.single_logits),
        coooda_cpp::nn::log_softmax_reference(inputs.single_logits),
        coooda_cpp::nn::cross_entropy_loss_reference(inputs.single_logits, inputs.single_logits.size() / 2),
        coooda_cpp::nn::mean_cross_entropy_loss_reference(
            inputs.batch_logits,
            inputs.targets,
            inputs.class_count
        ),
    };
}

void require_close(const std::string &label, float expected, float actual) {
    if (coooda_core::test::close_enough(expected, actual, kAbsTol, kRelTol)) {
        return;
    }

    std::ostringstream message;
    message << label << " expected " << expected << ", actual " << actual;
    coooda_core::fail(message.str());
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

void warm_up_cuda() {
    const std::vector<float> logits{1.0f, 2.0f, 3.0f, 4.0f};
    (void)coooda_cuda::nn::softmax_baseline(logits);
    (void)coooda_cuda::nn::log_softmax_baseline(logits);
    (void)coooda_cuda::nn::cross_entropy_loss_baseline(logits, 2);
    (void)coooda_cuda::nn::mean_cross_entropy_loss_baseline(logits, {1, 0}, 2);
    (void)coooda_cuda::nn::mean_cross_entropy_loss_device_reduce(logits, {1, 0}, 2);
}

void run_softmax_loss_compare_case() {
    const SoftmaxLossInputs inputs = make_inputs();
    const SoftmaxLossExpected expected = make_expected(inputs);
    warm_up_cuda();

    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_softmax_loss_softmax_baseline", [&]() {
        require_vector_close(
            "compare_softmax_loss_softmax_baseline",
            expected.softmax,
            coooda_cuda::nn::softmax_baseline(inputs.single_logits)
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_softmax_loss_log_softmax_baseline", [&]() {
        require_vector_close(
            "compare_softmax_loss_log_softmax_baseline",
            expected.log_softmax,
            coooda_cuda::nn::log_softmax_baseline(inputs.single_logits)
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_softmax_loss_cross_entropy_baseline", [&]() {
        require_close(
            "compare_softmax_loss_cross_entropy_baseline",
            expected.cross_entropy,
            coooda_cuda::nn::cross_entropy_loss_baseline(inputs.single_logits, inputs.single_logits.size() / 2)
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_softmax_loss_mean_cross_entropy_baseline", [&]() {
        require_close(
            "compare_softmax_loss_mean_cross_entropy_baseline",
            expected.mean_cross_entropy,
            coooda_cuda::nn::mean_cross_entropy_loss_baseline(
                inputs.batch_logits,
                inputs.targets,
                inputs.class_count
            )
        );
    }));
    coooda_core::bench::print_result(coooda_core::bench::time_once("compare_softmax_loss_mean_cross_entropy_device_reduce", [&]() {
        require_close(
            "compare_softmax_loss_mean_cross_entropy_device_reduce",
            expected.mean_cross_entropy,
            coooda_cuda::nn::mean_cross_entropy_loss_device_reduce(
                inputs.batch_logits,
                inputs.targets,
                inputs.class_count
            )
        );
    }));
}

} // namespace

namespace coooda_bench::compare {

void append_softmax_loss_compare_cases(std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    cases.push_back({
        "softmax_loss",
        "compare C++ reference, CUDA baseline, and CUDA device-reduced softmax/loss paths",
        []() { run_softmax_loss_compare_case(); },
    });
}

} // namespace coooda_bench::compare
