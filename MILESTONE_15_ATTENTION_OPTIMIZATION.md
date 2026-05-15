# Milestone 15: Attention Optimization

## Deliverable

At least one optimized attention variant that beats the baseline for a named shape family.

## Implement

- `coooda::nn::AttentionVariant`
- `coooda::nn::AttentionVariantConfig`
- `coooda::nn::select_attention_variant(...)`
- `coooda::kernels::attention_fused_mask_softmax(...)`
- `coooda::nn::attention_forward_fused_mask_softmax(...)`
- `coooda::kernels::attention_online_softmax(...)`
- `coooda::nn::attention_forward_online_softmax(...)`
- Optional: `coooda::kernels::attention_tiled(...)`
- `coooda::bench::estimate_attention_memory_bytes(...)`
- `coooda::bench::benchmark_attention_variants(...)`

## Steps

1. Re-run Milestone 14 tests and benchmarks.
2. Pick named shape families to optimize.
3. Implement fused mask plus softmax.
4. Run the full Milestone 14 correctness suite.
5. Compare timing and memory traffic against exact baseline shapes.
6. Implement online softmax.
7. Add tiled attention only if it is runnable and measured; otherwise write it as a non-counting note.
8. Capture Nsight evidence for the winning variant.

## Done

- A named optimized variant improves one named shape family.
- The improvement uses same-hardware, same-build comparison.
- `notes/attention_optimization.md` includes timing, memory estimate, and profiler evidence.
