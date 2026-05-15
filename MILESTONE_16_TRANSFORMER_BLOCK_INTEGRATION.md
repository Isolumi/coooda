# Milestone 16: Transformer Block Integration

## Deliverable

A reusable pre-norm decoder Transformer block.

## Implement

- `coooda::nn::TransformerBlockConfig`
- `coooda::nn::TransformerBlock`
- `coooda::tests::transformer_block_forward_cpu(...)`
- `coooda::nn::residual_add(...)`
- `coooda::nn::pre_norm_attention_forward(...)`
- `coooda::nn::pre_norm_ffn_forward(...)`
- `coooda::nn::transformer_block_forward(...)`
- `coooda::bench::benchmark_transformer_block_forward(...)`

## Steps

1. Define hidden size, heads, MLP expansion, norm type, and parameter names.
2. Assemble pre-norm attention path from existing modules.
3. Assemble pre-norm feed-forward path from existing MLP modules.
4. Add residual connections.
5. Test one tiny block against CPU or deterministic reference.
6. Test shape invariants, batch size 1, and context length 1.
7. Benchmark time by submodule.

## Done

- Block forward works through reusable modules.
- No private monolithic Transformer-block kernel exists.
- `notes/transformer_block.md` shows timing by attention, MLP, norm, and residual work.
