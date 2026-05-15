# Milestone 09: Normalization Kernels

## Deliverable

LayerNorm and RMSNorm modules.

## Implement

- `coooda::kernels::NormShape`
- `coooda::nn::LayerNormParams`
- `coooda::nn::RmsNormParams`
- `coooda::tests::layer_norm_cpu(...)`
- `coooda::tests::rms_norm_cpu(...)`
- `coooda::kernels::layer_norm_baseline(...)`
- `coooda::kernels::layer_norm_fused(...)`
- `coooda::kernels::rms_norm_baseline(...)`
- `coooda::kernels::rms_norm_fused(...)`
- `coooda::nn::LayerNorm`
- `coooda::nn::RmsNorm`

## Steps

1. Define row layout and epsilon policy.
2. Implement CPU LayerNorm.
3. Implement baseline CUDA LayerNorm.
4. Test zero variance, batch size 1, and hidden sizes around warp boundaries.
5. Benchmark hidden-size sweeps.
6. Implement RMSNorm.
7. Add fused variants and compare.

## Done

- LayerNorm and RMSNorm match CPU references.
- Fused variants are benchmarked.
- `notes/normalization.md` names the chosen Transformer norm.
