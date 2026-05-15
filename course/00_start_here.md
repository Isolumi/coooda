# 00: Start Here

Coooda is built in tiny checkpoints. Each topic keeps the C++ and CUDA work together.

## Loop

1. Write the C++ reference.
2. Run the C++ test.
3. Write the CUDA baseline.
4. Run the CUDA test.
5. Run the compare test.
6. Optimize CUDA only after correctness passes.

## Commands

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --no-tests=error
```

Benchmark runners accept `--list` and `--case <name>`. Filtered CTest commands use `--no-tests=error` so a missing checkpoint test fails instead of passing silently.

## Package Map

- `packages/coooda_cpp`: readable C++ reference backend.
- `packages/coooda_cuda`: CUDA backend.
- `packages/coooda_core`: shared infrastructure.
- `packages/coooda_compare`: backend equivalence checks.

## Rule

Do not add private one-off code to apps. Apps use library modules.
