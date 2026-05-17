#include <coooda_core/test_harness.hpp>
#include <coooda_core/tensor.hpp>
#include <coooda_core/status.hpp>

int main() {
    return coooda_core::test::run_tests({
        {"core_error_message", []() {
            const coooda_core::Error error("something failed");

            coooda_core::test::require_equal("something failed", error.what(), "error message");
        }},

        {"core_fail_throws_error", []() {
            bool threw = false;
            try {
                coooda_core::fail("expected failure");
            } catch (const coooda_core::Error &error) {
                threw = true;
                coooda_core::test::require_equal("expected failure", error.what(), "fail message");
            }

            coooda_core::test::require(threw, "fail should throw coooda_core::Error");
        }},

        {"shape_numel_multiplies_dimensions", []() {
            const coooda_core::Shape shape{{2, 3, 4}};
            coooda_core::test::require(coooda_core::numel(shape) == 24, "shahpe numel");
        }},

        {"shape_numel_empty_shape_is_zero", []() {
            const coooda_core::Shape shape{{}};
            coooda_core::test::require(coooda_core::numel(shape) == 0, "empty shape numel");
        }},

        {"shape_to_string_formats_dimensions", []() {
            const coooda_core::Shape shape{{2, 3, 4}};
            coooda_core::test::require_equal("[2x3x4]", coooda_core::to_string(shape), "shape string");
        }},

        {"shape_to_string_empty_shape", []() {
            const coooda_core::Shape shape{{}};
            coooda_core::test::require_equal("[]", coooda_core::to_string(shape), "empty shape string");
        }},
    });
}