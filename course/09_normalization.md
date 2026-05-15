# 09: Normalization

## Checkpoint 09.01: C++ Reference

Goal:
Implement C++ reference normalization functions.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/normalization.hpp`
- `packages/coooda_cpp/src/nn/normalization.cpp`
- `tests/cpp/test_normalization.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Reduction helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R normalization_cpp`

Pass:
- C++ normalization tests pass for known-value and seeded cases.

## Checkpoint 09.02: CUDA Baseline

Goal:
Implement matching CUDA normalization baselines.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/normalization.cuh`
- `packages/coooda_cuda/src/nn/normalization.cu`
- `tests/cuda/test_normalization.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA reduction patterns
- Device buffer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R normalization_cuda`

Pass:
- CUDA normalization tests match expected small cases.

## Checkpoint 09.03: Compare Backends

Goal:
Prove CUDA normalization matches the C++ reference.

You write:
- `tests/compare/test_normalization_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R normalization_compare`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 09.04: Optimize Or Integrate

Goal:
Add normalization to the release compare benchmark runner.

You write:
- `bench/compare/bench_normalization.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R normalization_compare`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case normalization`

Pass:
- The debug compare test passes, then the benchmark runs.
