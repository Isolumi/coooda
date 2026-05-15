# 16: Tiny GPT

## Checkpoint 16.01: C++ Reference

Goal:
Implement the C++ reference Tiny GPT model path and app entry.

You write:
- `packages/coooda_cpp/include/coooda_cpp/models/tiny_gpt.hpp`
- `packages/coooda_cpp/src/models/tiny_gpt.cpp`
- `apps/tiny_gpt_cpp.cpp`
- `tests/cpp/test_tiny_gpt.cpp`

Already provided:
- Transformer block helpers
- Embedding helpers
- Test harness

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R tiny_gpt_cpp`

Pass:
- C++ Tiny GPT tests pass for deterministic small inputs.

## Checkpoint 16.02: CUDA Baseline

Goal:
Implement the CUDA Tiny GPT model path and app entry.

You write:
- `packages/coooda_cuda/include/coooda_cuda/models/tiny_gpt.cuh`
- `packages/coooda_cuda/src/models/tiny_gpt.cu`
- `apps/tiny_gpt_cuda.cu`
- `tests/cuda/test_tiny_gpt.cu`

Already provided:
- CUDA transformer block helpers
- CUDA embedding helpers
- Device buffer helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R tiny_gpt_cuda`

Pass:
- CUDA Tiny GPT tests match expected small cases.

## Checkpoint 16.03: Compare Backends

Goal:
Prove CUDA Tiny GPT outputs match the C++ reference.

You write:
- `tests/compare/test_tiny_gpt_compare.cu`

Already provided:
- Compare helpers
- Seeded input helpers

Run:
- `cmake --build --preset debug`
- `ctest --preset debug -R tiny_gpt_compare`

Pass:
- Compare tests report no mismatches within the chosen tolerance.

## Checkpoint 16.04: Tiny GPT App Integration

Goal:
Run Tiny GPT through the app and release compare surfaces.

You write:
- `apps/tiny_gpt_cpp.cpp`
- `apps/tiny_gpt_cuda.cu`
- `bench/compare/bench_tiny_gpt.cu`

Already provided:
- App target wiring
- Benchmark timer helpers
- Release preset

Run:
- `cmake --build --preset release`
- `./build/release/apps/tiny_gpt_cpp`
- `./build/release/apps/tiny_gpt_cuda`
- `./build/release/bench/coooda_compare_bench --case tiny_gpt`

Pass:
- Apps run and the benchmark runs after debug correctness checks pass.
