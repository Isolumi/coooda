# Coooda Course Redesign Design

## Purpose

Restructure `coooda` from a verbose CUDA-first milestone list into a guided starter project for building two parallel Transformer libraries: one pure C++ reference library and one CUDA library. The course should support short, testable work loops for an ADHD-friendly learning cadence, while still ending in modular, reusable libraries and runnable Tiny GPT examples.

The user already understands Transformer math, so the course should avoid math exposition and focus on implementation order, API boundaries, verification, and performance iteration.

## Architecture

The repo will be a CMake workspace with separate packages for shared infrastructure, C++ implementation, CUDA implementation, and comparison/integration:

```text
coooda/
  CMakeLists.txt
  cmake/
  packages/
    coooda_core/
    coooda_cpp/
    coooda_cuda/
    coooda_compare/
  tests/
    cpp/
    cuda/
    compare/
  bench/
    cpp/
    cuda/
    compare/
  apps/
    inspect_device.cu
    run_checkpoint.cpp
    compare_backends.cu
    tiny_gpt_cpp.cpp
    tiny_gpt_cuda.cu
  course/
  notes/
```

`coooda_core` owns shared low-level support: shapes, tensor metadata contracts, status/error types, deterministic random fixtures, assertion helpers, mismatch reports, benchmark utilities, checkpoint serialization helpers, and CUDA runtime helpers that are not specific to a single CUDA operation.

`coooda_cpp` owns the pure C++ implementation. It is the first backend implemented for each operation and acts as the readable behavioral reference. It should compile and test without requiring a GPU.

`coooda_cuda` owns the CUDA implementation with a matching public API wherever practical. It includes baseline kernels first, then optimized kernels after correctness is locked.

`coooda_compare` owns backend equivalence and integration utilities. It calls into both `coooda_cpp` and `coooda_cuda`, compares outputs, reports mismatches, and provides shared runners for milestone checks. It does not hide the backend packages or become the main implementation location.

## Course Structure

Each learning unit lives in one `course/NN_topic.md` file. CPU and GPU work are intentionally kept together in the same file, because the intended learning loop is:

1. Build the C++ version to understand behavior.
2. Test the C++ version in isolation.
3. Translate the same behavior to CUDA.
4. Test CUDA against the C++ backend.
5. Optimize CUDA only after the comparison passes.

Every unit is split into small checkpoints. A checkpoint should be short enough to complete in one focused session and must end with an exact command and a concrete pass condition.

Checkpoint format:

```md
## Checkpoint 02.03: CUDA Vector Add Baseline

Goal:
Make CUDA vector add match the C++ reference.

You write:
- `packages/coooda_cuda/src/ops/vector_add.cu`

Already provided:
- Device allocation helpers
- Host/device copy helpers
- Random vector fixture
- Mismatch reporter

Run:
- `cmake --build build/debug --target coooda_tests`
- `ctest --test-dir build/debug -R vector_add_cuda_baseline --output-on-failure`

Pass:
- The CUDA baseline output matches `coooda_cpp::ops::vector_add`.
```

## Milestone Outline

The rewrite should replace the current broad milestone files with these single-file CPU-plus-CUDA learning units:

1. `00_start_here.md`: how the course works, how checkpoints work, how to run the scaffold.
2. `01_project_scaffold.md`: build targets, package layout, empty tests, empty runners.
3. `02_core_helpers.md`: status/error helpers, deterministic fixtures, mismatch reports, timers, CUDA checks, device inspection.
4. `03_memory_and_tensors.md`: host buffers, device buffers, tensor shape metadata, contiguous views, copy paths.
5. `04_vector_ops.md`: C++ vector ops, CUDA baseline kernels, grid-stride variants, first compare benchmark.
6. `05_reductions.md`: sum, max, argmax, dot product, shared-memory and warp-level CUDA variants.
7. `06_matmul.md`: C++ matmul, CUDA baseline, tiled CUDA matmul, variant selection.
8. `07_elementwise_and_fusion.md`: unary/binary ops, bias add, GELU/ReLU, fused CUDA variants.
9. `08_softmax_and_loss.md`: stable softmax, log softmax, cross entropy, fused CUDA loss path.
10. `09_normalization.md`: LayerNorm and RMSNorm in C++ and CUDA.
11. `10_embeddings.md`: token and position embeddings with shared sequence layout.
12. `11_linear_and_mlp.md`: parameters, Linear, activation, MLP forward.
13. `12_backward_and_optimizers.md`: manual gradients, finite difference checks, SGD, optional AdamW.
14. `13_tiny_mlp_training.md`: tiny dataset, one-batch overfit, checkpoint resume.
15. `14_attention.md`: materialized causal attention, fused mask-softmax, online softmax.
16. `15_transformer_block.md`: pre-norm decoder block assembled from modules.
17. `16_tiny_gpt.md`: Tiny GPT C++ and CUDA apps, training, checkpointing, generation, bottleneck report.

Each file should include both the CPU and CUDA checkpoints for that topic. There should not be separate CPU and CUDA syllabi.

## Starter Scaffold

The implementation pass should create runnable project infrastructure immediately, before asking the user to write operation code:

- Root CMake workspace with package targets.
- `coooda_core` library target.
- `coooda_cpp` library target.
- `coooda_cuda` library target.
- `coooda_compare` library target.
- `coooda_tests` executable with a tiny custom test harness.
- `coooda_bench` executable with shared benchmark helpers.
- `inspect_device`, `compare_backends`, `tiny_gpt_cpp`, and `tiny_gpt_cuda` app targets.
- Empty or stubbed module files for the full course path.
- CMake presets or documented configure commands for debug, release, and profile builds.

The scaffold should include boilerplate the user does not want to spend attention on: runners, timers, CUDA error checks, deterministic random input generation, output comparison, failure formatting, and benchmark summaries.

## API Boundary Rules

The C++ and CUDA packages should have parallel module names and matching public function names where that keeps comparison simple. Shared data contracts live in `coooda_core`, not copied into each backend.

Example:

```text
packages/coooda_cpp/include/coooda_cpp/ops/vector.hpp
packages/coooda_cuda/include/coooda_cuda/ops/vector.cuh
packages/coooda_compare/include/coooda_compare/ops/vector_compare.hpp
```

Model-level code should use public module APIs. Tiny GPT must not become a one-file special case with private kernels or private CPU code.

## Testing Strategy

Testing should be layered:

- C++ tests verify behavior without a GPU.
- CUDA tests verify baseline and optimized kernels against fixed expected cases and CPU references.
- Compare tests call both packages and report exact mismatches.
- Integration tests prove modules compose into MLP, attention, Transformer block, and Tiny GPT paths.

Every operation should include exact small cases, boundary cases, deterministic random cases with recorded seeds, and tolerance policies. Gradient checks should run only on tiny shapes.

## Benchmarking Strategy

Benchmarks are provided as runners, not learning tasks. The user should only write benchmark registration when a new operation needs a new shape family.

Benchmarking happens after correctness. Reports should include operation, backend, variant, shape, dtype, build mode, GPU, warmups, trials, median/min/max time, throughput or bandwidth, and speedup versus baseline.

## Documentation Tone

Course docs should be direct and action-oriented:

- No long math explanations.
- No boilerplate-heavy prose.
- Each checkpoint should answer what to write, what is already provided, what command to run, and what passing looks like.
- Each milestone should feel like several small wins rather than one large vague task.

## Success Criteria

The redesign is successful when:

- A fresh checkout can configure, build, and run empty tests immediately.
- The user can open any course file and see the CPU-to-CUDA progression in one place.
- Each checkpoint has one small observable outcome.
- C++ and CUDA implementations remain separate packages.
- Parent comparison tooling can prove backend equivalence.
- The final Tiny GPT exists in both C++ and CUDA forms and uses reusable library modules.
