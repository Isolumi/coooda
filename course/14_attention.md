# 14: Attention

## Checkpoint 14.01: C++ Reference

Goal:
Implement the C++ reference attention path.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/attention.hpp`
- `packages/coooda_cpp/src/nn/attention.cpp`
- `tests/cpp/test_attention.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Matmul helpers
- Softmax and normalization helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R attention_cpp --no-tests=error`

Pass:
- C++ attention tests pass for known-value and masked cases.

## Checkpoint 14.02: CUDA Baseline

Goal:
Implement a matching CUDA attention baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/attention.cuh`
- `packages/coooda_cuda/src/nn/attention.cu`
- `tests/cuda/test_attention.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA matmul helpers
- CUDA softmax helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R attention_cuda --no-tests=error`

Pass:
- CUDA attention tests match expected small cases.

## Checkpoint 14.03: Compare Backends

Goal:
Prove CUDA attention outputs match the C++ reference.

You write:
- `tests/compare/test_attention_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R attention_compare --no-tests=error`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 14.04: Optimize Or Integrate

Goal:
Add attention to the release compare benchmark runner.

You write:
- `bench/compare/bench_attention.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R attention_compare --no-tests=error`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case attention`

Pass:
- The debug compare test passes, then the benchmark runs.
