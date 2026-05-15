# 04: Vector Ops

## Checkpoint 04.01: C++ Reference

Goal:
Implement the C++ reference functions for vector add, SAXPY, ReLU, sigmoid, and elementwise multiply.

You write:
- `packages/coooda_cpp/include/coooda_cpp/ops/vector.hpp`
- `packages/coooda_cpp/src/ops/vector.cpp`
- `tests/cpp/test_vector_ops.cpp`

Already provided:
- Test harness
- Mismatch reporting
- Build target wiring

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R vector_ops_cpp`

Pass:
- The C++ vector operation tests pass.

## Checkpoint 04.02: CUDA Baseline

Goal:
Implement matching one-thread-per-element CUDA baselines.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh`
- `packages/coooda_cuda/src/ops/vector.cu`
- `tests/cuda/test_vector_ops.cu`

Already provided:
- CUDA error checks
- Device inspection
- Host comparison utilities

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R vector_ops_cuda`

Pass:
- CUDA vector operations match the expected small cases.

## Checkpoint 04.03: Compare Backends

Goal:
Prove CUDA matches the C++ reference.

You write:
- `packages/coooda_compare/include/coooda_compare/ops/vector_compare.hpp`
- `packages/coooda_compare/src/ops/vector_compare.cu`
- `tests/compare/test_vector_ops_compare.cu`

Already provided:
- Compare package target wiring
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R vector_ops_compare`

Pass:
- The compare test reports no mismatch for exact, boundary, and seeded random cases.

## Checkpoint 04.04: Grid-Stride CUDA Variant

Goal:
Add grid-stride CUDA variants and compare them against the baseline.

You write:
- `packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh`
- `packages/coooda_cuda/src/ops/vector.cu`
- `bench/cuda/bench_vector_ops.cu`
- `bench/compare/bench_vector_ops_compare.cu`

Already provided:
- Release preset
- Benchmark runner

Run:
- `cmake --build --preset release`
- `./build/release/bench/coooda_compare_bench --case vector_ops`

Pass:
- Correctness still passes before any speed claim.
