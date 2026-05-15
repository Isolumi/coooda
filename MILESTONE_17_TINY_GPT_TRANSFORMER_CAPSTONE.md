# Milestone 17: Tiny GPT Transformer Capstone

## Deliverable

Tiny GPT trains, checkpoints, reloads, and generates through the `coooda` library.

## Implement

- `coooda::models::TinyGptConfig`
- `coooda::models::TinyGptModel`
- `coooda::training::CharDataset`
- `coooda::training::encode_chars(...)`
- `coooda::training::decode_chars(...)`
- `coooda::training::make_lm_batches(...)`
- `coooda::models::tiny_gpt_forward(...)`
- `coooda::models::tiny_gpt_loss(...)`
- `coooda::models::tiny_gpt_train_step(...)`
- `coooda::models::generate_greedy(...)`
- `coooda::models::save_tiny_gpt_checkpoint(...)`
- `coooda::models::load_tiny_gpt_checkpoint(...)`
- `coooda::bench::benchmark_tiny_gpt_training_step(...)`
- `coooda::bench::benchmark_tiny_gpt_generation(...)`

## Steps

1. Freeze a tiny config: vocab, context, hidden size, heads, blocks, batch size.
2. Build a character dataset and language-model batches.
3. Assemble model from Embedding, TransformerBlock, norm, Linear, loss, optimizer.
4. Run one-batch overfit.
5. Train for a fixed number of steps.
6. Save and load a checkpoint.
7. Generate text from a fixed prompt.
8. Benchmark training tokens/sec and generation tokens/sec.
9. Write final API and bottleneck review.

## Done

- `apps/tiny_gpt` trains without one-off model-local kernels.
- Checkpoint reload preserves model behavior.
- Generation works from a fixed prompt.
- `notes/tiny_gpt_final.md` includes loss curve, sample output, timing, and remaining bottleneck.
