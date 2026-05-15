# 13: Tiny MLP Training

## Checkpoint 13.01: C++ Reference

Goal:
Implement a C++ reference tiny MLP training loop.

You write:
- `packages/coooda_cpp/include/coooda_cpp/training/tiny_mlp.hpp`
- `packages/coooda_cpp/src/training/tiny_mlp.cpp`
- `tests/cpp/test_tiny_mlp_training.cpp`

Already provided:
- Linear and MLP helpers
- Backward and optimizer helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R tiny_mlp_training_cpp`

Pass:
- C++ tiny MLP training tests pass for deterministic updates.

## Checkpoint 13.02: CUDA Baseline

Goal:
Implement a matching CUDA tiny MLP training loop.

You write:
- `packages/coooda_cuda/include/coooda_cuda/training/tiny_mlp.cuh`
- `packages/coooda_cuda/src/training/tiny_mlp.cu`
- `tests/cuda/test_tiny_mlp_training.cu`

Already provided:
- CUDA layer helpers
- CUDA optimizer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R tiny_mlp_training_cuda`

Pass:
- CUDA tiny MLP training tests match expected small cases.

## Checkpoint 13.03: Compare Backends

Goal:
Prove CUDA training updates match the C++ reference.

You write:
- `tests/compare/test_tiny_mlp_training_compare.cu`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R tiny_mlp_training_compare`

Pass:
- Compare tests report no mismatches for deterministic training steps.

## Checkpoint 13.04: Optimize Or Integrate

Goal:
Add tiny MLP training to the release compare benchmark runner.

You write:
- `bench/compare/bench_tiny_mlp_training.cu`

Already provided:
- Benchmark timer helpers
- Release preset

Run:
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case tiny_mlp_training`

Pass:
- The benchmark runs after debug correctness checks pass.
