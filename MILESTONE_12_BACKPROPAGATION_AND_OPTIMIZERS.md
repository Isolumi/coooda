# Milestone 12: Backpropagation And Optimizers

## Deliverable

Manual backward for the MLP path plus SGD.

## Implement

- `coooda::nn::GradBuffer`
- `coooda::nn::zero_grad(...)`
- `coooda::nn::accumulate_grad(...)`
- `coooda::tests::finite_difference_check(...)`
- `coooda::tests::linear_backward_cpu(...)`
- `coooda::tests::relu_backward_cpu(...)`
- `coooda::tests::softmax_cross_entropy_logits_backward_cpu(...)`
- `coooda::nn::linear_backward(...)`
- `coooda::nn::relu_backward(...)`
- `coooda::nn::softmax_cross_entropy_logits_backward(...)`
- `coooda::optim::SgdConfig`
- `coooda::optim::SgdOptimizer`
- `coooda::optim::sgd_step(...)`
- Optional: `coooda::optim::AdamWOptimizer`

## Steps

1. Define gradient storage.
2. Implement Linear backward CPU reference.
3. Implement Linear backward CUDA.
4. Verify with finite differences.
5. Implement activation backward.
6. Implement softmax-cross-entropy logits gradient.
7. Implement SGD.
8. Test repeated accumulation and zeroing.
9. Add optional AdamW only after SGD passes.

## Done

- Linear, activation, and loss gradients pass checks.
- SGD update matches CPU reference.
- `notes/backprop.md` records gradient-check tolerances.
