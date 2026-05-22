add_library(coooda_cuda
  src/memory/device_buffer.cu
  src/ops/elementwise.cu
  src/ops/matmul.cu
  src/ops/reductions.cu
  src/ops/vector.cu
  src/package.cu
)

target_include_directories(coooda_cuda PUBLIC include)
target_link_libraries(coooda_cuda PUBLIC coooda_core CUDA::cudart coooda_options)
