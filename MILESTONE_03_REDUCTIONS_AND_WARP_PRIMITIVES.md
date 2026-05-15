# Milestone 03: Reductions And Warp Primitives

## Deliverable

Reusable sum, max, argmax, and dot-product reductions.

## Implement

- `coooda::kernels::ArgMaxResult<T>`
- `coooda::kernels::ReductionConfig`
- `coooda::tests::reduce_sum_cpu(...)`
- `coooda::tests::reduce_max_cpu(...)`
- `coooda::tests::argmax_cpu(...)`
- `coooda::tests::dot_product_cpu(...)`
- `coooda::kernels::reduce_sum_baseline(...)`
- `coooda::kernels::reduce_sum_shared(...)`
- `coooda::kernels::reduce_sum_warp(...)`
- `coooda::kernels::reduce_max_baseline(...)`
- `coooda::kernels::reduce_max_shared(...)`
- `coooda::kernels::reduce_max_warp(...)`
- `coooda::kernels::argmax_baseline(...)`
- `coooda::kernels::argmax_warp(...)`
- `coooda::kernels::dot_product_baseline(...)`
- `coooda::kernels::dot_product_warp(...)`

## Steps

1. Define result shapes, especially argmax tie behavior.
2. Implement CPU references.
3. Implement `reduce_sum_baseline`.
4. Test exact small arrays, empty policy, negatives, and random arrays.
5. Benchmark baseline sum.
6. Add shared-memory sum.
7. Add warp-shuffle sum.
8. Repeat the pattern for max, argmax, and dot product.

## Done

- All reductions match CPU references.
- Baseline, shared, and warp timings are compared.
- `notes/reductions.md` says which reduction style later softmax/norm kernels should use.
