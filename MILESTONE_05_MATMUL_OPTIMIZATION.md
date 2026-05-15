# Milestone 05: Matmul Optimization

## Deliverable

Tiled matmul that keeps the Milestone 04 API but improves at least one useful shape.

## Implement

- `coooda::kernels::MatmulVariant`
- `coooda::kernels::MatmulTileConfig`
- `coooda::kernels::matmul_tiled(...)`
- `coooda::kernels::matmul_register_blocked(...)`
- `coooda::kernels::select_matmul_variant(...)`
- `coooda::bench::benchmark_matmul_variants(...)`
- `coooda::bench::print_matmul_variant_table(...)`

## Steps

1. Re-run Milestone 04 tests and benchmarks.
2. Implement one shared-memory tiled variant.
3. Test it against every Milestone 04 correctness case.
4. Benchmark exact baseline shapes against tiled shapes.
5. Sweep tile sizes.
6. Capture one Nsight Compute note.
7. Add register blocking only after tiled matmul is correct and measured.

## Done

- Tiled matmul preserves public behavior.
- At least one named shape improves over baseline.
- `notes/matmul_optimization.md` includes timing table and profiler evidence.
