#include <coooda_core/test_harness.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace {

bool contains(const std::string &text, const std::string &needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace

int main() {
    return coooda_core::test::run_tests({
        {"core_compare_matching_vectors", []() {
             const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
                 "core_compare",
                 std::vector<float>{1.0f, 2.0f, 3.0f},
                 std::vector<float>{1.0f, 2.0f, 3.0f},
                 0.0f,
                 0.0f
             );

             coooda_core::test::require(report.matched, "matching vectors should match");
             coooda_core::test::require_equal("core_compare: matched", report.to_string(), "matched report");
         }},

        {"core_compare_uses_tolerance", []() {
             const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
                 "core_compare_tolerance",
                 std::vector<float>{100.0f},
                 std::vector<float>{100.01f},
                 0.02f,
                 0.0f
             );

             coooda_core::test::require(report.matched, "values inside tolerance should match");
         }},

        {"core_compare_reports_value_mismatch", []() {
             const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
                 "core_compare_mismatch",
                 std::vector<float>{1.0f, 2.0f, 3.0f},
                 std::vector<float>{1.0f, 2.5f, 3.0f},
                 0.01f,
                 0.0f
             );

             coooda_core::test::require(!report.matched, "different vectors should mismatch");
             coooda_core::test::require(report.index == 1, "mismatch index");
             coooda_core::test::require(report.expected == 2.0f, "mismatch expected value");
             coooda_core::test::require(report.actual == 2.5f, "mismatch actual value");
             coooda_core::test::require(report.absolute_difference == 0.5f, "mismatch difference");
             coooda_core::test::require(report.tolerance == 0.01f, "mismatch tolerance");

             const std::string message = report.to_string();
             coooda_core::test::require(
                 contains(message, "core_compare_mismatch: mismatch at index 1"),
                 "mismatch report should include label and index"
             );
             coooda_core::test::require(
                 contains(message, "expected 2"),
                 "mismatch report should include expected value"
             );
             coooda_core::test::require(
                 contains(message, "actual 2.5"),
                 "mismatch report should include actual value"
             );
         }},

        {"core_compare_reports_size_mismatch", []() {
             const coooda_core::test::MismatchReport report = coooda_core::test::compare_vectors(
                 "core_compare_size",
                 std::vector<float>{1.0f, 2.0f},
                 std::vector<float>{1.0f},
                 0.01f,
                 0.0f
             );

             coooda_core::test::require(!report.matched, "different sizes should mismatch");
             coooda_core::test::require(report.index == 1, "size mismatch index");
             coooda_core::test::require(report.expected == 2.0f, "size mismatch expected value");
             coooda_core::test::require(std::isnan(report.actual), "missing actual value should be NaN");
             coooda_core::test::require(
                 std::isinf(report.absolute_difference),
                 "size mismatch difference should be infinite"
             );
         }},
    });
}
