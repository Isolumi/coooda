# Milestone 11: Linear Layers And MLP

## Deliverable

Forward-only Linear and MLP modules assembled from existing kernels.

## Implement

- `coooda::nn::Parameter`
- `coooda::nn::ParameterInit`
- `coooda::nn::initialize_xavier(...)`
- `coooda::nn::initialize_zeros(...)`
- `coooda::nn::LinearConfig`
- `coooda::nn::Linear`
- `coooda::tests::linear_forward_cpu(...)`
- `coooda::nn::linear_forward(...)`
- `coooda::nn::ActivationKind`
- `coooda::nn::MlpConfig`
- `coooda::nn::Mlp`
- `coooda::tests::mlp_forward_cpu(...)`
- `coooda::nn::mlp_forward(...)`

## Steps

1. Define parameter ownership.
2. Implement initialization.
3. Implement Linear using matmul plus bias.
4. Test known small weights, zero bias, and batch size 1.
5. Implement MLP as Linear, activation, Linear.
6. Benchmark Linear and MLP forward.
7. Add fused bias plus activation if Milestone 07 supports it.

## Done

- MLP uses modules, not private kernels.
- Forward results match CPU references.
- `notes/mlp_forward.md` records timing.
