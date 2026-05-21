#include <coooda_core/status.hpp>
#include <coooda_core/test_harness.hpp>
#include <coooda_cpp/ops/reductions.hpp>
#include <coooda_cuda/ops/reductions.cuh>

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

template <typename Fn>
bool throws_core_error(Fn &&fn) {
    try {
        fn();
    } catch (const coooda_core::Error &) {
        return true;
    }
    return false;
}

void compare_non_empty_case(const std::string &label, const std::vector<float> &values) {
    require_close(
        label + "_sum",
        coooda_cpp::ops::sum_reference(values),
        coooda_cuda::ops::sum_baseline(values)
    );
    require_close(
        label + "_device_sum",
        coooda_cpp::ops::sum_reference(values),
        coooda_cuda::ops::sum_device_reduce(values)
    );
    require_close(
        label + "_max",
        coooda_cpp::ops::max_reference(values),
        coooda_cuda::ops::max_baseline(values)
    );
    require_close(
        label + "_device_max",
        coooda_cpp::ops::max_reference(values),
        coooda_cuda::ops::max_device_reduce(values)
    );
    require_close(
        label + "_mean",
        coooda_cpp::ops::mean_reference(values),
        coooda_cuda::ops::mean_baseline(values)
    );
    require_close(
        label + "_device_mean",
        coooda_cpp::ops::mean_reference(values),
        coooda_cuda::ops::mean_device_reduce(values)
    );
    coooda_core::test::require(
        coooda_cpp::ops::argmax_reference(values) == coooda_cuda::ops::argmax_baseline(values),
        label + "_argmax"
    );
    coooda_core::test::require(
        coooda_cpp::ops::argmax_reference(values) == coooda_cuda::ops::argmax_device_reduce(values),
        label + "_device_argmax"
    );
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"reductions_compare_empty", []() {
             require_close(
                 "empty_sum",
                 coooda_cpp::ops::sum_reference({}),
                 coooda_cuda::ops::sum_baseline({})
             );
             require_close(
                 "empty_device_sum",
                 coooda_cpp::ops::sum_reference({}),
                 coooda_cuda::ops::sum_device_reduce({})
             );
             coooda_core::test::require(
                 throws_core_error([]() { (void)coooda_cpp::ops::max_reference({}); }) ==
                     throws_core_error([]() { (void)coooda_cuda::ops::max_baseline({}); }),
                 "empty max throw behavior should match"
             );
             coooda_core::test::require(
                 throws_core_error([]() { (void)coooda_cpp::ops::max_reference({}); }) ==
                     throws_core_error([]() { (void)coooda_cuda::ops::max_device_reduce({}); }),
                 "empty device max throw behavior should match"
             );
             coooda_core::test::require(
                 throws_core_error([]() { (void)coooda_cpp::ops::mean_reference({}); }) ==
                     throws_core_error([]() { (void)coooda_cuda::ops::mean_baseline({}); }),
                 "empty mean throw behavior should match"
             );
             coooda_core::test::require(
                 throws_core_error([]() { (void)coooda_cpp::ops::mean_reference({}); }) ==
                     throws_core_error([]() { (void)coooda_cuda::ops::mean_device_reduce({}); }),
                 "empty device mean throw behavior should match"
             );
             coooda_core::test::require(
                 throws_core_error([]() { (void)coooda_cpp::ops::argmax_reference({}); }) ==
                     throws_core_error([]() { (void)coooda_cuda::ops::argmax_baseline({}); }),
                 "empty argmax throw behavior should match"
             );
             coooda_core::test::require(
                 throws_core_error([]() { (void)coooda_cpp::ops::argmax_reference({}); }) ==
                     throws_core_error([]() { (void)coooda_cuda::ops::argmax_device_reduce({}); }),
                 "empty device argmax throw behavior should match"
             );
         }},

        {"reductions_compare_small", []() {
             compare_non_empty_case("small", {1.0f, -2.0f, 3.5f, 0.5f});
         }},

        {"reductions_compare_single", []() {
             compare_non_empty_case("single", {-7.25f});
         }},

        {"reductions_compare_argmax_tie", []() {
             compare_non_empty_case("argmax_tie", {1.0f, 4.0f, 4.0f, 3.0f});
         }},

        {"reductions_compare_seeded", []() {
             compare_non_empty_case(
                 "seeded",
                 coooda_core::test::seeded_vector(1024, 0x5A5AU, -10.0f, 10.0f)
             );
         }},
    });
}
