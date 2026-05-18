#include <coooda_core/host_buffer.hpp>
#include <coooda_core/tensor.hpp>
#include <coooda_core/test_harness.hpp>

int main() {
    return coooda_core::test::run_tests({
        {"host_buffer_default_is_empty", []() {
             const coooda_core::HostBuffer buffer;

             coooda_core::test::require(buffer.empty(), "default host buffer should be empty");
             coooda_core::test::require(buffer.size() == 0, "default host buffer size");
         }},

        {"host_buffer_allocates_storage", []() {
             coooda_core::HostBuffer buffer(4);

             coooda_core::test::require(!buffer.empty(), "allocated host buffer should not be empty");
             coooda_core::test::require(buffer.size() == 4, "allocated host buffer size");
             coooda_core::test::require(buffer.data() != nullptr, "allocated host buffer data pointer");
         }},

        {"host_buffer_reads_and_writes_values", []() {
             coooda_core::HostBuffer buffer(3);

             buffer[0] = 1.0f;
             buffer[1] = 2.0f;
             buffer[2] = 3.0f;

             coooda_core::test::require(buffer[0] == 1.0f, "host buffer value 0");
             coooda_core::test::require(buffer[1] == 2.0f, "host buffer value 1");
             coooda_core::test::require(buffer[2] == 3.0f, "host buffer value 2");
         }},

        {"tensor_allocates_storage_from_shape", []() {
             coooda_core::Tensor tensor({{2, 3}});

             coooda_core::test::require(tensor.size() == 6, "tensor storage size");
             coooda_core::test::require(!tensor.empty(), "tensor should not be empty");
             coooda_core::test::require(tensor.data() != nullptr, "tensor data pointer");
             coooda_core::test::require_equal("[2x3]", coooda_core::to_string(tensor.shape()), "tensor shape");
         }},

        {"tensor_reads_and_writes_values", []() {
             coooda_core::Tensor tensor({{2, 2}});

             tensor[0] = 1.0f;
             tensor[1] = 2.0f;
             tensor[2] = 3.0f;
             tensor[3] = 4.0f;

             coooda_core::test::require(tensor[0] == 1.0f, "tensor value 0");
             coooda_core::test::require(tensor[1] == 2.0f, "tensor value 1");
             coooda_core::test::require(tensor[2] == 3.0f, "tensor value 2");
             coooda_core::test::require(tensor[3] == 4.0f, "tensor value 3");
         }},

        {"tensor_view_references_same_storage", []() {
             coooda_core::Tensor tensor({{2, 2}});
             tensor[1] = 7.0f;

             coooda_core::TensorView view = tensor.view();

             coooda_core::test::require(view.size() == tensor.size(), "tensor view size");
             coooda_core::test::require(view.data() == tensor.data(), "tensor view data pointer");
             coooda_core::test::require_equal("[2x2]", coooda_core::to_string(view.shape()), "tensor view shape");
             coooda_core::test::require(view[1] == 7.0f, "tensor view reads tensor value");

             view[2] = 9.0f;
             coooda_core::test::require(tensor[2] == 9.0f, "tensor view writes tensor value");
         }},
    });
}
