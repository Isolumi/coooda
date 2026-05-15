# Milestone 14: Attention Baseline

## Deliverable

Correct materialized scaled dot-product attention.

## Implement

- `coooda::nn::AttentionConfig`
- `coooda::nn::AttentionLayout`
- `coooda::tests::attention_forward_cpu(...)`
- `coooda::nn::MultiHeadAttention`
- `coooda::nn::project_qkv(...)`
- `coooda::kernels::attention_scores_baseline(...)`
- `coooda::kernels::apply_causal_mask_baseline(...)`
- `coooda::kernels::attention_weight_values_baseline(...)`
- `coooda::nn::attention_forward_baseline(...)`
- `coooda::bench::benchmark_attention_baseline(...)`

## Steps

1. Define batch, head, sequence, and head-dimension layout.
2. Reuse Linear or matmul for Q, K, V.
3. Implement CPU one-head causal attention.
4. Implement score matrix, scale, mask, softmax, and value weighting.
5. Test hand-checkable one-head cases.
6. Test causal masking, sequence length 1, head dimension 1, and multi-head layout.
7. Benchmark across batch, heads, sequence length, and head dimension.

## Done

- Baseline attention matches CPU references.
- Stage timing is recorded.
- `notes/attention_baseline.md` explains the main cost.
