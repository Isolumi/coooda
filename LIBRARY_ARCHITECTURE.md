# Library Architecture

Use `coooda` as the package root and C++ namespace.

## Layers

- `coooda::core`: CUDA checks, device info, timers
- `coooda::memory`: device and pinned host buffers
- `coooda::tensor`: shapes, strides, tensors, views
- `coooda::kernels`: CUDA kernels and variants
- `coooda::nn`: modules such as Linear, MLP, Attention, TransformerBlock
- `coooda::optim`: SGD and optional AdamW
- `coooda::training`: batches, metrics, checkpoints, train loops
- `coooda::models`: TinyMlp and TinyGpt
- `coooda::tests`: CPU references and validation helpers
- `coooda::bench`: benchmark helpers

## Rule

CUDA kernels live in `coooda::kernels`. CPU references live in `coooda::tests`. Modules in `coooda::nn` and `coooda::models` call public APIs; they should not hide private one-off kernels.
