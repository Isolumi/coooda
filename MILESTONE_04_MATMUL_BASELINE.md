# Milestone 04: Matmul Baseline

## Deliverable

Correct baseline matmul and matvec.

## Implement

- `coooda::kernels::MatmulShape`
- `coooda::kernels::MatvecShape`
- `coooda::kernels::validate_matmul_shape(...)`
- `coooda::tests::matmul_cpu(...)`
- `coooda::tests::matvec_cpu(...)`
- `coooda::kernels::matmul_baseline(...)`
- `coooda::kernels::matvec_baseline(...)`
- `coooda::bench::estimate_matmul_flops(...)`
- `coooda::bench::benchmark_matmul_baseline(...)`

## Steps

1. Define row-major layout.
2. Implement shape validation.
3. Implement `matmul_cpu`.
4. Implement one-output-element-per-thread `matmul_baseline`.
5. Test known small matrices.
6. Test square, rectangular, single row, single column, and non-multiple dimensions.
7. Benchmark GFLOP/s for several shapes.
8. Implement and test `matvec_baseline`.

## Done

- Baseline matmul is correct before any tiling.
- Baseline GFLOP/s is recorded.
- `notes/matmul_baseline.md` explains the bottleneck.
