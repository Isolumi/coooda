# 10: Embeddings

## Checkpoint 10.01: C++ Reference

Goal:
Implement the C++ reference embedding lookup path.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/embedding.hpp`
- `packages/coooda_cpp/src/nn/embedding.cpp`
- `tests/cpp/test_embeddings.cpp`
- `tests/CMakeLists.txt`

Already provided:
- Tensor helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_cpp --no-tests=error`

Pass:
- C++ embedding tests pass for known token IDs and boundary checks.

## Checkpoint 10.02: CUDA Baseline

Goal:
Implement the CUDA embedding lookup baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/embedding.cuh`
- `packages/coooda_cuda/src/nn/embedding.cu`
- `tests/cuda/test_embeddings.cu`
- `tests/CMakeLists.txt`

Already provided:
- Device buffer helpers
- CUDA checks

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_cuda --no-tests=error`

Pass:
- CUDA embedding tests match expected small cases.

## Checkpoint 10.03: Compare Backends

Goal:
Prove CUDA embedding lookup matches the C++ reference.

You write:
- `tests/compare/test_embeddings_compare.cu`
- `tests/CMakeLists.txt`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_compare --no-tests=error`

Pass:
- Compare tests report no mismatches for selected token batches.

## Checkpoint 10.04: Optimize Or Integrate

Goal:
Add embeddings to the release compare benchmark runner.

You write:
- `bench/compare/bench_embeddings.cu`
- `bench/CMakeLists.txt`

Already provided:
- Benchmark timer helpers

Run:
- `cmake --preset debug`
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_compare --no-tests=error`
- `cmake --preset release`
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case embeddings`

Pass:
- The debug compare test passes, then the benchmark runs.
