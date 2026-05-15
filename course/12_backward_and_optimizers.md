# 12: Backward And Optimizers

## Checkpoint 12.01: C++ Reference

Goal:
Implement C++ reference backward helpers and SGD updates.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/backward.hpp`
- `packages/coooda_cpp/include/coooda_cpp/optim/sgd.hpp`
- `packages/coooda_cpp/src/nn/backward.cpp`
- `packages/coooda_cpp/src/optim/sgd.cpp`
- `tests/cpp/test_backward_optimizers.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Layer helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R backward_optimizers_cpp --no-tests=error`

Pass:
- C++ backward and optimizer tests pass for known-value updates.

## Checkpoint 12.02: CUDA Baseline

Goal:
Implement matching CUDA backward helpers and SGD updates.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/backward.cuh`
- `packages/coooda_cuda/include/coooda_cuda/optim/sgd.cuh`
- `packages/coooda_cuda/src/nn/backward.cu`
- `packages/coooda_cuda/src/optim/sgd.cu`
- `tests/cuda/test_backward_optimizers.cu`
- `tests/CMakeLists.txt`

Already provided:
- Device buffer helpers
- CUDA layer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R backward_optimizers_cuda --no-tests=error`

Pass:
- CUDA backward and optimizer tests match expected small cases.

## Checkpoint 12.03: Compare Backends

Goal:
Prove CUDA gradients and updates match the C++ reference.

You write:
- `tests/compare/test_backward_optimizers_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R backward_optimizers_compare --no-tests=error`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 12.04: Optimize Or Integrate

Goal:
Add backward and optimizer cases to the release compare benchmark runner.

You write:
- `bench/compare/bench_backward_optimizers.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R backward_optimizers_compare --no-tests=error`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case backward_optimizers`

Pass:
- The debug compare test passes, then the benchmark runs.
