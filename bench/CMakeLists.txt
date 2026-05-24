add_executable(coooda_cpp_bench cpp/bench_cpp_smoke.cpp)
target_link_libraries(coooda_cpp_bench PRIVATE coooda_core coooda_cpp)

add_executable(coooda_cuda_bench
  cuda/bench_cuda_smoke.cu
  cuda/bench_vector_ops.cu
)
target_link_libraries(coooda_cuda_bench PRIVATE coooda_core coooda_cuda)

add_executable(coooda_compare_bench
  compare/bench_compare_smoke.cu
  compare/bench_core_helpers.cu
  compare/bench_elementwise.cu
  compare/bench_matmul.cu
  compare/bench_memory_tensors.cu
  compare/bench_normalization.cu
  compare/bench_reductions.cu
  compare/bench_softmax_loss.cu
  compare/bench_vector_ops_compare.cu
)
target_link_libraries(coooda_compare_bench PRIVATE coooda_core coooda_compare coooda_cpp coooda_cuda)
