# Milestone 10: Embeddings And Positional Data

## Deliverable

Token and position embeddings that produce Transformer-ready sequence tensors.

## Implement

- `coooda::nn::EmbeddingConfig`
- `coooda::nn::SequenceLayout`
- `coooda::nn::validate_token_ids(...)`
- `coooda::tests::embedding_lookup_cpu(...)`
- `coooda::tests::add_positional_embedding_cpu(...)`
- `coooda::nn::Embedding`
- `coooda::nn::PositionEmbedding`
- `coooda::nn::embedding_lookup_baseline(...)`
- `coooda::nn::add_positional_embedding_baseline(...)`
- `coooda::bench::benchmark_embedding_lookup(...)`

## Steps

1. Choose layout: batch, sequence, hidden.
2. Define invalid token behavior.
3. Implement CPU embedding lookup.
4. Implement CUDA embedding lookup.
5. Test repeated IDs, first ID, last ID, invalid ID, batch size 1, and context length 1.
6. Add positional embeddings.
7. Benchmark lookup bandwidth.

## Done

- Embedding output layout is documented.
- Token and position paths pass tests.
- `notes/embeddings.md` explains the layout used by attention.
