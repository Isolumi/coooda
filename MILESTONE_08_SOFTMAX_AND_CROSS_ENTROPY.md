# Milestone 08: Softmax And Cross Entropy

## Deliverable

Stable row-wise softmax and cross entropy for training.

## Implement

- `coooda::kernels::SoftmaxShape`
- `coooda::nn::LossReduction`
- `coooda::tests::softmax_cpu(...)`
- `coooda::tests::log_softmax_cpu(...)`
- `coooda::tests::cross_entropy_cpu(...)`
- `coooda::kernels::softmax_baseline(...)`
- `coooda::kernels::log_softmax_baseline(...)`
- `coooda::nn::cross_entropy_baseline(...)`
- `coooda::nn::softmax_cross_entropy_fused(...)`
- `coooda::bench::benchmark_softmax_rows(...)`

## Steps

1. Define logits shape and target format.
2. Implement stable CPU softmax.
3. Implement baseline CUDA softmax.
4. Test large positive, large negative, equal, and one-class logits.
5. Benchmark row-length sweeps.
6. Implement cross entropy.
7. Implement fused softmax cross entropy.
8. Compare fused versus unfused.

## Done

- Adversarial numeric tests pass.
- Loss values match CPU references.
- `notes/softmax_cross_entropy.md` records stability and timing.
