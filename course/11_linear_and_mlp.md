# 11: Linear And MLP

## Checkpoint 11.01: C++ Reference

Goal:
Implement C++ reference linear and MLP layers.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/linear.hpp`
- `packages/coooda_cpp/include/coooda_cpp/nn/mlp.hpp`
- `packages/coooda_cpp/src/nn/linear.cpp`
- `packages/coooda_cpp/src/nn/mlp.cpp`
- `tests/cpp/test_linear_mlp.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Matmul helpers
- Elementwise helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R linear_mlp_cpp`

Pass:
- C++ linear and MLP tests pass for known-value and seeded cases.

## Checkpoint 11.02: CUDA Baseline

Goal:
Implement matching CUDA linear and MLP baselines.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/linear.cuh`
- `packages/coooda_cuda/include/coooda_cuda/nn/mlp.cuh`
- `packages/coooda_cuda/src/nn/linear.cu`
- `packages/coooda_cuda/src/nn/mlp.cu`
- `tests/cuda/test_linear_mlp.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA matmul helpers
- Device buffer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R linear_mlp_cuda`

Pass:
- CUDA linear and MLP tests match expected small cases.

## Checkpoint 11.03: Compare Backends

Goal:
Prove CUDA linear and MLP outputs match the C++ reference.

You write:
- `tests/compare/test_linear_mlp_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R linear_mlp_compare`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 11.04: Optimize Or Integrate

Goal:
Add linear and MLP cases to the release compare benchmark runner.

You write:
- `bench/compare/bench_linear_mlp.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R linear_mlp_compare`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case linear_mlp`

Pass:
- The debug compare test passes, then the benchmark runs.
