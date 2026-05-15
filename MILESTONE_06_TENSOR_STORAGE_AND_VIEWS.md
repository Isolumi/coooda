# Milestone 06: Tensor Storage And Views

## Deliverable

Basic tensor metadata and views that earlier kernels can consume.

## Implement

- `coooda::tensor::Shape`
- `coooda::tensor::Strides`
- `coooda::tensor::make_contiguous_strides(...)`
- `coooda::tensor::numel(...)`
- `coooda::tensor::is_contiguous(...)`
- `coooda::tensor::Tensor<T>`
- `coooda::tensor::TensorView<T>`
- `coooda::tensor::as_view(...)`
- `coooda::tensor::slice(...)`
- `coooda::tensor::require_contiguous(...)`
- Tensor overloads for `vector_add`, `relu`, and one reduction or matmul path

## Steps

1. Implement `Shape` and `Strides`.
2. Test rank, extents, element count, and contiguous strides.
3. Implement `TensorView<T>`.
4. Implement owning `Tensor<T>` backed by `DeviceBuffer<T>`.
5. Implement host/device copy helpers for tensors.
6. Add tensor-view overloads for vector add and ReLU.
7. Add one tensor-view reduction or matmul overload.
8. Reject unsupported non-contiguous views clearly.

## Done

- Tensor copy round trips pass.
- Existing kernels can be called through tensor views.
- `notes/tensor_views.md` lists which kernels require contiguous tensors.
