# 15: Transformer Block

## Checkpoint 15.01: C++ Reference

Goal:
Implement the C++ reference transformer block.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/transformer_block.hpp`
- `packages/coooda_cpp/src/nn/transformer_block.cpp`
- `tests/cpp/test_transformer_block.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Attention helpers
- Linear and MLP helpers
- Normalization helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R transformer_block_cpp`

Pass:
- C++ transformer block tests pass for known-value and seeded cases.

## Checkpoint 15.02: CUDA Baseline

Goal:
Implement a matching CUDA transformer block baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/transformer_block.cuh`
- `packages/coooda_cuda/src/nn/transformer_block.cu`
- `tests/cuda/test_transformer_block.cu`
- `tests/CMakeLists.txt`

Already provided:
- CUDA attention helpers
- CUDA linear and MLP helpers
- CUDA normalization helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R transformer_block_cuda`

Pass:
- CUDA transformer block tests match expected small cases.

## Checkpoint 15.03: Compare Backends

Goal:
Prove CUDA transformer block outputs match the C++ reference.

You write:
- `tests/compare/test_transformer_block_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R transformer_block_compare`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 15.04: Optimize Or Integrate

Goal:
Add transformer block cases to the release compare benchmark runner.

You write:
- `bench/compare/bench_transformer_block.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R transformer_block_compare`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case transformer_block`

Pass:
- The debug compare test passes, then the benchmark runs.
