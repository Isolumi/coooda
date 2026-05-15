# 07: Elementwise And Fusion

## Checkpoint 07.01: C++ Reference

Goal:
Implement C++ reference elementwise operations used by fused paths.

You write:
- `packages/coooda_cpp/include/coooda_cpp/ops/elementwise.hpp`
- `packages/coooda_cpp/src/ops/elementwise.cpp`
- `tests/cpp/test_elementwise.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Vector operation patterns
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R elementwise_cpp`

Pass:
- C++ elementwise tests pass for unary, binary, and broadcast-like cases.

## Checkpoint 07.02: CUDA Baseline

Goal:
Implement matching CUDA elementwise baselines.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/elementwise.cuh`
- `packages/coooda_cuda/src/ops/elementwise.cu`
- `tests/cuda/test_elementwise.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA vector operation patterns
- Device buffer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R elementwise_cuda`

Pass:
- CUDA elementwise tests match expected small cases.

## Checkpoint 07.03: Compare Backends

Goal:
Prove CUDA elementwise operations match the C++ reference.

You write:
- `tests/compare/test_elementwise_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R elementwise_compare`

Pass:
- Compare tests report no mismatches for baseline and fused cases.

## Checkpoint 07.04: Optimize Or Integrate

Goal:
Add elementwise and fused cases to the release compare benchmark runner.

You write:
- `bench/compare/bench_elementwise.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R elementwise_compare`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case elementwise`

Pass:
- The debug compare test passes, then the benchmark runs.
