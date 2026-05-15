# 06: Matmul

## Checkpoint 06.01: C++ Reference

Goal:
Implement the C++ reference matrix multiply path.

You write:
- `packages/coooda_cpp/include/coooda_cpp/ops/matmul.hpp`
- `packages/coooda_cpp/src/ops/matmul.cpp`
- `tests/cpp/test_matmul.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Tensor shape helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R matmul_cpp`

Pass:
- C++ matmul tests pass for shape checks and known-value inputs.

## Checkpoint 06.02: CUDA Baseline

Goal:
Implement a simple CUDA matmul baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/matmul.cuh`
- `packages/coooda_cuda/src/ops/matmul.cu`
- `tests/cuda/test_matmul.cu`
- `tests/CMakeLists.txt`

Already provided:
- Device buffer helpers
- CUDA checks

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R matmul_cuda`

Pass:
- CUDA matmul tests match expected small matrices.

## Checkpoint 06.03: Compare Backends

Goal:
Prove CUDA matmul matches the C++ reference.

You write:
- `tests/compare/test_matmul_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R matmul_compare`

Pass:
- Compare tests report no mismatches across the selected matrix sizes.

## Checkpoint 06.04: Optimize Or Integrate

Goal:
Add matmul to the release compare benchmark runner.

You write:
- `bench/compare/bench_matmul.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R matmul_compare`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case matmul`

Pass:
- The debug compare test passes, then the benchmark runs.
