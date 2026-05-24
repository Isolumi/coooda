#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/loss.hpp>
#include <coooda_cuda/nn/loss.cuh>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-4f;
constexpr float kRelTol = 1.0e-5f;

void require_close(const std::string &label, float expected, float actual) {
    if (coooda_core::test::close_enough(expected, actual, kAbsTol, kRelTol)) {
        return;
    }

    std::ostringstream message;
    message << label << " expected " << expected << ", actual " << actual;
    coooda_core::test::require(false, message.str());
}

void require_vector_close(
    const std::string &label,
    const std::vector<float> &expected,
    const std::vector<float> &actual
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

void compare_single_case(
    const std::string &label,
    const std::vector<float> &logits,
    std::size_t target_index
) {
    require_vector_close(
        label + "_softmax_baseline",
        coooda_cpp::nn::softmax_reference(logits),
        coooda_cuda::nn::softmax_baseline(logits)
    );
    require_vector_close(
        label + "_log_softmax_baseline",
        coooda_cpp::nn::log_softmax_reference(logits),
        coooda_cuda::nn::log_softmax_baseline(logits)
    );
    require_close(
        label + "_cross_entropy_baseline",
        coooda_cpp::nn::cross_entropy_loss_reference(logits, target_index),
        coooda_cuda::nn::cross_entropy_loss_baseline(logits, target_index)
    );
}

void compare_batch_case(
    const std::string &label,
    const std::vector<float> &logits,
    const std::vector<std::size_t> &targets,
    std::size_t class_count
) {
    const float expected = coooda_cpp::nn::mean_cross_entropy_loss_reference(logits, targets, class_count);
    require_close(
        label + "_mean_cross_entropy_baseline",
        expected,
        coooda_cuda::nn::mean_cross_entropy_loss_baseline(logits, targets, class_count)
    );
    require_close(
        label + "_mean_cross_entropy_device_reduce",
        expected,
        coooda_cuda::nn::mean_cross_entropy_loss_device_reduce(logits, targets, class_count)
    );
}

std::vector<float> seeded(std::size_t size, std::uint32_t seed) {
    return coooda_core::test::seeded_vector(size, seed, -8.0f, 8.0f);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"softmax_loss_compare_known_values", []() {
             compare_single_case("known", {1.0f, 2.0f, 3.0f}, 2);
             compare_batch_case(
                 "known_batch",
                 {1.0f, 2.0f, 3.0f, 2.0f, 0.0f, -1.0f},
                 {2, 0},
                 3
             );
         }},

        {"softmax_loss_compare_large_logits", []() {
             compare_single_case("large", {1000.0f, 1001.0f, 1002.0f}, 1);
             compare_batch_case(
                 "large_batch",
                 {1000.0f, 1001.0f, 1002.0f, 900.0f, 899.0f, 898.0f},
                 {2, 0},
                 3
             );
         }},

        {"softmax_loss_compare_seeded_single", []() {
             compare_single_case("seeded_single", seeded(513, 0x5017U), 257);
         }},

        {"softmax_loss_compare_seeded_batch", []() {
             compare_batch_case("seeded_batch", seeded(24, 0xA11CEU), {0, 3, 5, 1}, 6);
             compare_batch_case("wide_seeded_batch", seeded(17 * 13, 0xB0BU), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3}, 13);
         }},
    });
}
