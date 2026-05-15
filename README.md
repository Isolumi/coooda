# Coooda

Coooda is a guided CUDA/C++ course that builds two parallel Transformer libraries:

- `coooda_cpp`: a pure C++ reference implementation.
- `coooda_cuda`: a CUDA implementation with baseline and optimized kernels.
- `coooda_compare`: runners that compare both backends.
- `coooda_core`: shared shapes, test helpers, benchmark helpers, CUDA checks, and reporting.

The course is organized as small checkpoints. Each topic keeps the CPU and CUDA work in the same file so the flow is always:

1. Write the C++ reference.
2. Test it.
3. Translate it to CUDA.
4. Compare CUDA against C++.
5. Optimize only after correctness passes.

## Quick Start

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/apps/inspect_device
./build/debug/apps/run_checkpoint
./build/debug/apps/compare_backends
```

Start at `course/00_start_here.md`.

## Project Layout

```text
packages/coooda_core/      shared infrastructure
packages/coooda_cpp/       C++ backend
packages/coooda_cuda/      CUDA backend
packages/coooda_compare/   backend comparison utilities
tests/                     CTest smoke and checkpoint tests
bench/                     benchmark runners
apps/                      runnable demos and final Tiny GPT apps
course/                    checkpoint guide
```
