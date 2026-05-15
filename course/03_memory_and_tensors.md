# 03: Memory And Tensors

## Checkpoint 03.01: C++ Reference

Goal:
Add host buffer and tensor behavior needed by C++ reference operators.

You write:
- `packages/coooda_core/include/coooda_core/host_buffer.hpp`
- `packages/coooda_core/include/coooda_core/tensor.hpp`
- `packages/coooda_core/src/tensor.cpp`
- `tests/cpp/test_memory_tensors.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Core status type
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R memory_tensors_cpp --no-tests=error`

Pass:
- Host buffer and tensor tests pass for allocation, size, and view cases.

## Checkpoint 03.02: CUDA Baseline

Goal:
Add a device buffer wrapper for allocation and host-device copies.

You write:
- `packages/coooda_cuda/include/coooda_cuda/memory/device_buffer.cuh`
- `packages/coooda_cuda/src/memory/device_buffer.cu`
- `tests/cuda/test_memory_tensors.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA check helpers
- Device inspection helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R memory_tensors_cuda --no-tests=error`

Pass:
- Device buffer tests pass for allocation, copy in, copy out, and empty cases.

## Checkpoint 03.03: Compare Backends

Goal:
Confirm host and device tensor transfers preserve values.

You write:
- `tests/compare/test_memory_tensors_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Existing smoke examples
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R memory_tensors_compare --no-tests=error`

Pass:
- Round-trip compare tests report no mismatches.

## Checkpoint 03.04: Optimize Or Integrate

Goal:
Add memory transfer coverage to the release compare benchmark runner.

You write:
- `bench/compare/bench_memory_tensors.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R memory_tensors_compare --no-tests=error`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case memory_tensors`

Pass:
- The debug compare test passes, then the benchmark runs.
