# 08: Softmax And Loss

## Checkpoint 08.01: C++ Reference

Goal:
Implement C++ reference softmax and loss functions.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/loss.hpp`
- `packages/coooda_cpp/src/nn/loss.cpp`
- `tests/cpp/test_softmax_loss.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Reduction helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R softmax_loss_cpp`

Pass:
- C++ softmax and loss tests pass for known-value and seeded cases.

## Checkpoint 08.02: CUDA Baseline

Goal:
Implement matching CUDA softmax and loss baselines.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/loss.cuh`
- `packages/coooda_cuda/src/nn/loss.cu`
- `tests/cuda/test_softmax_loss.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA reduction patterns
- Device buffer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R softmax_loss_cuda`

Pass:
- CUDA softmax and loss tests match expected small cases.

## Checkpoint 08.03: Compare Backends

Goal:
Prove CUDA softmax and loss outputs match the C++ reference.

You write:
- `tests/compare/test_softmax_loss_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R softmax_loss_compare`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 08.04: Optimize Or Integrate

Goal:
Add softmax and loss cases to the release compare benchmark runner.

You write:
- `bench/compare/bench_softmax_loss.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R softmax_loss_compare`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case softmax_loss`

Pass:
- The debug compare test passes, then the benchmark runs.
