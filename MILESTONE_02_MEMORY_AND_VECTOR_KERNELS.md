# Milestone 02: Memory And Vector Kernels

## Deliverable

Device memory utilities plus baseline and optimized vector kernels.

## Implement

- `coooda::memory::DeviceBuffer<T>`
- `coooda::memory::PinnedHostBuffer<T>`
- `coooda::memory::copy_host_to_device(...)`
- `coooda::memory::copy_device_to_host(...)`
- `coooda::memory::copy_device_to_device(...)`
- `coooda::tests::vector_add_cpu(...)`
- `coooda::tests::saxpy_cpu(...)`
- `coooda::tests::relu_cpu(...)`
- `coooda::tests::sigmoid_cpu(...)`
- `coooda::tests::elementwise_multiply_cpu(...)`
- `coooda::kernels::vector_add_baseline(...)`
- `coooda::kernels::vector_add_grid_stride(...)`
- `coooda::kernels::saxpy_baseline(...)`
- `coooda::kernels::saxpy_grid_stride(...)`
- `coooda::kernels::relu_baseline(...)`
- `coooda::kernels::relu_grid_stride(...)`
- `coooda::kernels::sigmoid_baseline(...)`
- `coooda::kernels::sigmoid_grid_stride(...)`
- `coooda::kernels::elementwise_multiply_baseline(...)`
- `coooda::kernels::elementwise_multiply_grid_stride(...)`

## Steps

1. Implement `DeviceBuffer<T>`.
2. Implement copy helpers.
3. Implement `vector_add_cpu`.
4. Implement `vector_add_baseline`.
5. Test size 0, size 1, non-block-multiple, and large vector.
6. Benchmark baseline vector add bandwidth.
7. Add `vector_add_grid_stride` and compare.
8. Repeat the same baseline then grid-stride pattern for SAXPY, ReLU, sigmoid, and multiply.
9. Run one pinned-memory transfer benchmark.

## Done

- Every CUDA vector result matches its CPU reference.
- Baseline and grid-stride timings are recorded.
- `notes/vector_kernels.md` says which version is faster and why.
