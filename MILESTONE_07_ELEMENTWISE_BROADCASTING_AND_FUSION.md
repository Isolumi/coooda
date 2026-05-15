# Milestone 07: Elementwise, Broadcasting, And Fusion

## Deliverable

Reusable tensor elementwise operations plus one fused operation.

## Implement

- `coooda::kernels::UnaryOp`
- `coooda::kernels::BinaryOp`
- `coooda::kernels::BroadcastPlan`
- `coooda::kernels::make_broadcast_plan(...)`
- `coooda::tests::elementwise_unary_cpu(...)`
- `coooda::tests::elementwise_binary_cpu(...)`
- `coooda::tests::bias_add_cpu(...)`
- `coooda::kernels::elementwise_unary_baseline(...)`
- `coooda::kernels::elementwise_binary_baseline(...)`
- `coooda::kernels::bias_add_baseline(...)`
- `coooda::kernels::fused_bias_relu(...)`
- `coooda::kernels::fused_bias_gelu(...)`

## Steps

1. Define broadcasting rules.
2. Implement CPU references.
3. Implement contiguous unary ops.
4. Implement contiguous binary ops.
5. Add scalar, row, and column broadcast tests.
6. Implement bias add.
7. Benchmark bias add plus activation as separate kernels.
8. Implement fused bias plus activation and compare.

## Done

- Broadcasting behavior is explicit.
- Fused and unfused paths produce the same values.
- `notes/elementwise_fusion.md` shows whether fusion helped.
