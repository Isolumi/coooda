# 02: Core Helpers

## Checkpoint 02.01: C++ Reference

Goal:
Build the status and tensor helpers used by later C++ reference code.

You write:
- `packages/coooda_core/include/coooda_core/status.hpp`
- `packages/coooda_core/include/coooda_core/tensor.hpp`
- `packages/coooda_core/src/status.cpp`
- `packages/coooda_core/src/tensor.cpp`
- `tests/cpp/test_core_helpers.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Existing package examples
- Test harness entry points

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R core_helpers_cpp --no-tests=error`

Pass:
- Core helper tests pass for success, failure, shape, and storage cases.

## Checkpoint 02.02: CUDA Baseline

Goal:
Add CUDA check and device helpers for later kernels.

You write:
- `packages/coooda_core/include/coooda_core/cuda_check.cuh`
- `packages/coooda_core/include/coooda_core/device.hpp`
- `packages/coooda_core/src/cuda_check.cu`
- `packages/coooda_core/src/device.cu`
- `tests/cuda/test_core_cuda.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA runtime linkage
- Device inspection app target

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R core_cuda --no-tests=error`

Pass:
- CUDA helper tests report errors cleanly and find a usable device when available.

## Checkpoint 02.03: Compare Backends

Goal:
Prove shared result and mismatch reporting works for both backends.

You write:
- `tests/compare/test_core_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Existing smoke examples
- Mismatch formatting helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R core_compare --no-tests=error`

Pass:
- Compare helper tests pass for matching and mismatching inputs.

## Checkpoint 02.04: Optimize Or Integrate

Goal:
Wire core helpers into the release compare benchmark runner.

You write:
- `bench/compare/bench_core_helpers.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R core_compare --no-tests=error`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case core_helpers`

Pass:
- The debug compare test passes, then the benchmark runs.
