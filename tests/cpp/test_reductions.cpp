#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/reductions.hpp>

#include <cstddef>
#include <vector>

namespace {

void require_close(float expected, float actual, const std::string &message) {
    coooda_core::test::require(
        coooda_core::test::close_enough(expected, actual, 1.0e-6f, 1.0e-6f),
        message
    );
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

float manual_sum(const std::vector<float> &values) {
    float total = 0.0f;
    for (float value : values) {
        total += value;
    }
    return total;
}

float manual_max(const std::vector<float> &values) {
    float best = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > best) {
            best = values[i];
        }
    }
    return best;
}

std::size_t manual_argmax(const std::vector<float> &values) {
    std::size_t best_index = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best_index]) {
            best_index = i;
        }
    }
    return best_index;
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"reductions_cpp_empty_inputs", []() {
             coooda_core::test::require(
                 coooda_cpp::ops::sum_reference({}) == 0.0f,
                 "empty sum should be zero"
             );
             require_core_error(
                 []() { (void)coooda_cpp::ops::max_reference({}); },
                 "empty max should fail"
             );
             require_core_error(
                 []() { (void)coooda_cpp::ops::mean_reference({}); },
                 "empty mean should fail"
             );
             require_core_error(
                 []() { (void)coooda_cpp::ops::argmax_reference({}); },
                 "empty argmax should fail"
             );
         }},

        {"reductions_cpp_small_inputs", []() {
             const std::vector<float> values{1.0f, -2.0f, 3.5f, 0.5f};

             require_close(3.0f, coooda_cpp::ops::sum_reference(values), "small sum");
             require_close(3.5f, coooda_cpp::ops::max_reference(values), "small max");
             require_close(0.75f, coooda_cpp::ops::mean_reference(values), "small mean");
             coooda_core::test::require(
                 coooda_cpp::ops::argmax_reference(values) == 2,
                 "small argmax"
             );
         }},

        {"reductions_cpp_single_value", []() {
             const std::vector<float> values{-7.25f};

             require_close(-7.25f, coooda_cpp::ops::sum_reference(values), "single sum");
             require_close(-7.25f, coooda_cpp::ops::max_reference(values), "single max");
             require_close(-7.25f, coooda_cpp::ops::mean_reference(values), "single mean");
             coooda_core::test::require(
                 coooda_cpp::ops::argmax_reference(values) == 0,
                 "single argmax"
             );
         }},

        {"reductions_cpp_argmax_returns_first_tie", []() {
             const std::vector<float> values{1.0f, 4.0f, 4.0f, 3.0f};

             coooda_core::test::require(
                 coooda_cpp::ops::argmax_reference(values) == 1,
                 "argmax should return first max index"
             );
         }},

        {"reductions_cpp_seeded_inputs", []() {
             const std::vector<float> values =
                 coooda_core::test::seeded_vector(129, 0x5A5AU, -10.0f, 10.0f);
             const float expected_sum = manual_sum(values);

             require_close(expected_sum, coooda_cpp::ops::sum_reference(values), "seeded sum");
             require_close(manual_max(values), coooda_cpp::ops::max_reference(values), "seeded max");
             require_close(
                 expected_sum / static_cast<float>(values.size()),
                 coooda_cpp::ops::mean_reference(values),
                 "seeded mean"
             );
             coooda_core::test::require(
                 coooda_cpp::ops::argmax_reference(values) == manual_argmax(values),
                 "seeded argmax"
             );
         }},
    });
}
