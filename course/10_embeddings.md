# 10: Embeddings

## Checkpoint 10.01: C++ Reference

Goal:
Implement the C++ reference embedding lookup path.

You write:
- `packages/coooda_cpp/include/coooda_cpp/nn/embedding.hpp`
- `packages/coooda_cpp/src/nn/embedding.cpp`
- `tests/cpp/test_embeddings.cpp`

Already provided:
- Tensor helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_cpp`

Pass:
- C++ embedding tests pass for known token IDs and boundary checks.

## Checkpoint 10.02: CUDA Baseline

Goal:
Implement the CUDA embedding lookup baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/nn/embedding.cuh`
- `packages/coooda_cuda/src/nn/embedding.cu`
- `tests/cuda/test_embeddings.cu`

Already provided:
- Device buffer helpers
- CUDA checks

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_cuda`

Pass:
- CUDA embedding tests match expected small cases.

## Checkpoint 10.03: Compare Backends

Goal:
Prove CUDA embedding lookup matches the C++ reference.

You write:
- `tests/compare/test_embeddings_compare.cu`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R embeddings_compare`

Pass:
- Compare tests report no mismatches for selected token batches.

## Checkpoint 10.04: Optimize Or Integrate

Goal:
Add embeddings to the release compare benchmark runner.

You write:
- `bench/compare/bench_embeddings.cu`

Already provided:
- Benchmark timer helpers
- Release preset

Run:
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case embeddings`

Pass:
- The benchmark runs after debug correctness checks pass.
