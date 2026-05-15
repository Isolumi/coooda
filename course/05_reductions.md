# 05: Reductions

## Checkpoint 05.01: C++ Reference

Goal:
Implement C++ reference reductions for sum, max, mean, and argmax.

You write:
- `packages/coooda_cpp/include/coooda_cpp/ops/reductions.hpp`
- `packages/coooda_cpp/src/ops/reductions.cpp`
- `tests/cpp/test_reductions.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Tensor helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R reductions_cpp --no-tests=error`

Pass:
- C++ reduction tests pass for empty, small, and seeded inputs.

## Checkpoint 05.02: CUDA Baseline

Goal:
Implement CUDA reduction baselines with simple block-level kernels.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/reductions.cuh`
- `packages/coooda_cuda/src/ops/reductions.cu`
- `tests/cuda/test_reductions.cu`
- `tests/CMakeLists.txt`

Already provided:
- Device buffer helpers
- CUDA checks

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R reductions_cuda --no-tests=error`

Pass:
- CUDA reduction tests match expected values for small inputs.

## Checkpoint 05.03: Compare Backends

Goal:
Prove CUDA reductions match the C++ reference.

You write:
- `tests/compare/test_reductions_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R reductions_compare --no-tests=error`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 05.04: Optimize Or Integrate

Goal:
Add reductions to the release compare benchmark runner.

You write:
- `bench/compare/bench_reductions.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R reductions_compare --no-tests=error`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case reductions`

Pass:
- The debug compare test passes, then the benchmark runs.
