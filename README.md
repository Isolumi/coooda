# Coooda CUDA Transformer Course

Build a small CUDA/C++ library named `coooda`, ending with a tiny GPT-style decoder-only Transformer.

This course is intentionally not a textbook. Each milestone gives you a small deliverable, the exact API names to implement, and the order to build them. You write all code yourself.

## Rules

- Package root: `coooda/`
- C++ namespace: `coooda`
- No PyTorch, TensorFlow, JAX, Eigen, Thrust, cuBLAS, or cuDNN for implementation.
- Each kernel family gets a CPU reference, baseline CUDA version, optimized CUDA version, tests, benchmark, and short note.
- Build a library first. The final Transformer should use library modules, not one giant program.

## Suggested Project Shape

```text
coooda/
  include/coooda/
  src/
  tests/
  bench/
  apps/
  notes/
```

Keep the markdown course files separate from your implementation code.

## How To Use A Milestone

For each milestone:

1. Read `Deliverable`.
2. Create the items listed in `Implement`.
3. Follow `Steps` in order.
4. Stop only when every item in `Done` passes.

## Milestones

- `MILESTONE_00_PROJECT_SETUP_AND_CMAKE.md`
- `MILESTONE_01_ENVIRONMENT_AND_MEASUREMENT.md`
- `MILESTONE_02_MEMORY_AND_VECTOR_KERNELS.md`
- `MILESTONE_03_REDUCTIONS_AND_WARP_PRIMITIVES.md`
- `MILESTONE_04_MATMUL_BASELINE.md`
- `MILESTONE_05_MATMUL_OPTIMIZATION.md`
- `MILESTONE_06_TENSOR_STORAGE_AND_VIEWS.md`
- `MILESTONE_07_ELEMENTWISE_BROADCASTING_AND_FUSION.md`
- `MILESTONE_08_SOFTMAX_AND_CROSS_ENTROPY.md`
- `MILESTONE_09_NORMALIZATION_KERNELS.md`
- `MILESTONE_10_EMBEDDINGS_AND_POSITIONAL_DATA.md`
- `MILESTONE_11_LINEAR_LAYERS_AND_MLP.md`
- `MILESTONE_12_BACKPROPAGATION_AND_OPTIMIZERS.md`
- `MILESTONE_13_TINY_NEURAL_NETWORK_TRAINING.md`
- `MILESTONE_14_ATTENTION_BASELINE.md`
- `MILESTONE_15_ATTENTION_OPTIMIZATION.md`
- `MILESTONE_16_TRANSFORMER_BLOCK_INTEGRATION.md`
- `MILESTONE_17_TINY_GPT_TRANSFORMER_CAPSTONE.md`

## Final Result

You are done when `apps/tiny_gpt` can train a tiny model, save/load a checkpoint, generate text, and report where time is spent.
