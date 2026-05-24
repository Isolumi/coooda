#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/nn/loss.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTol = 1.0e-5f;
constexpr float kRelTol = 1.0e-5f;

void require_close(float expected, float actual, const std::string &label) {
    coooda_core::test::require(
        coooda_core::test::close_enough(expected, actual, kAbsTol, kRelTol),
        label + " expected " + std::to_string(expected) + ", actual " + std::to_string(actual)
    );
}

void require_vector_close(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const std::string &label
) {
    const coooda_core::test::MismatchReport report =
        coooda_core::test::compare_vectors(label, expected, actual, kAbsTol, kRelTol);
    coooda_core::test::require(report.matched, report.to_string());
}

template <typename Fn>
void require_core_error(Fn &&fn, const std::string &message) {
    bool threw = false;
    try {
        fn();
    } catch (const coooda_core::Error &) {
        threw = true;
    }
    coooda_core::test::require(threw, message);
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"softmax_loss_cpp_softmax_known_values", []() {
             const std::vector<float> probs = coooda_cpp::nn::softmax_reference({1.0f, 2.0f, 3.0f});
             require_vector_close({0.0900306f, 0.244728f, 0.665241f}, probs, "softmax known values");
             require_close(1.0f, probs[0] + probs[1] + probs[2], "softmax sum");
             coooda_core::test::require(probs[2] > probs[1] && probs[1] > probs[0], "softmax ordering");
         }},

        {"softmax_loss_cpp_log_softmax_known_values", []() {
             require_vector_close(
                 {-2.407606f, -1.407606f, -0.407606f},
                 coooda_cpp::nn::log_softmax_reference({1.0f, 2.0f, 3.0f}),
                 "log softmax known values"
             );
         }},

        {"softmax_loss_cpp_cross_entropy_known_values", []() {
             require_close(
                 0.407606f,
                 coooda_cpp::nn::cross_entropy_loss_reference({1.0f, 2.0f, 3.0f}, 2),
                 "cross entropy target 2"
             );
             require_close(
                 2.407606f,
                 coooda_cpp::nn::cross_entropy_loss_reference({1.0f, 2.0f, 3.0f}, 0),
                 "cross entropy target 0"
             );
         }},

        {"softmax_loss_cpp_mean_cross_entropy_known_values", []() {
             require_close(
                 0.288726f,
                 coooda_cpp::nn::mean_cross_entropy_loss_reference(
                     {1.0f, 2.0f, 3.0f, 2.0f, 0.0f, -1.0f},
                     {2, 0},
                     3
                 ),
                 "mean cross entropy"
             );
         }},

        {"softmax_loss_cpp_stable_for_large_logits", []() {
             const std::vector<float> probs =
                 coooda_cpp::nn::softmax_reference({1000.0f, 1001.0f, 1002.0f});
             require_vector_close({0.0900306f, 0.244728f, 0.665241f}, probs, "stable softmax");
             for (const float value : probs) {
                 coooda_core::test::require(std::isfinite(value), "softmax should stay finite");
             }
         }},

        {"softmax_loss_cpp_seeded_batch_loss_is_finite", []() {
             const std::vector<float> logits =
                 coooda_core::test::seeded_vector(24, 0x5017U, -8.0f, 8.0f);
             const float loss =
                 coooda_cpp::nn::mean_cross_entropy_loss_reference(logits, {0, 3, 5, 1}, 6);
             coooda_core::test::require(std::isfinite(loss), "seeded mean loss should be finite");
             coooda_core::test::require(loss > 0.0f, "seeded mean loss should be positive");
         }},

        {"softmax_loss_cpp_rejects_invalid_inputs", []() {
             require_core_error(
                 []() { (void)coooda_cpp::nn::softmax_reference({}); },
                 "softmax should reject empty logits"
             );
             require_core_error(
                 []() { (void)coooda_cpp::nn::cross_entropy_loss_reference({1.0f, 2.0f}, 2); },
                 "cross entropy should reject out-of-range target"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::mean_cross_entropy_loss_reference(
                         {1.0f, 2.0f, 3.0f},
                         {0, 1},
                         2
                     );
                 },
                 "mean cross entropy should reject mismatched batch shape"
             );
             require_core_error(
                 []() {
                     (void)coooda_cpp::nn::mean_cross_entropy_loss_reference(
                         {1.0f, 2.0f},
                         {2},
                         2
                     );
                 },
                 "mean cross entropy should reject out-of-range target"
             );
         }},
    });
}
