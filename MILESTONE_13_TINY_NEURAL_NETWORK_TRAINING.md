# Milestone 13: Tiny Neural Network Training

## Deliverable

A small MLP trains end to end before attention begins.

## Implement

- `coooda::training::Dataset`
- `coooda::training::Batch`
- `coooda::training::DataLoader`
- `coooda::training::TrainConfig`
- `coooda::training::TrainState`
- `coooda::training::TrainingMetrics`
- `coooda::training::Checkpoint`
- `coooda::tests::make_tiny_classification_dataset(...)`
- `coooda::training::make_dataloader(...)`
- `coooda::models::TinyMlpConfig`
- `coooda::models::TinyMlp`
- `coooda::training::train_step(...)`
- `coooda::training::evaluate(...)`
- `coooda::training::save_checkpoint(...)`
- `coooda::training::load_checkpoint(...)`

## Steps

1. Create a fixed-seed tiny classification dataset.
2. Build `TinyMlp` from existing Linear and activation modules.
3. Run one-batch overfit.
4. Train for a fixed number of steps.
5. Require a documented validation loss or accuracy threshold.
6. Save and reload a checkpoint.
7. Compare resumed training against uninterrupted training.
8. Benchmark one full training step.

## Done

- One-batch overfit passes.
- Fixed training task reaches the threshold.
- Checkpoint resume matches uninterrupted training within tolerance.
- `notes/tiny_mlp_training.md` records loss curve and timing.
