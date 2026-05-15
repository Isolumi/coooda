add_library(coooda_cuda
  src/ops/vector.cu
  src/package.cu
)

target_include_directories(coooda_cuda PUBLIC include)
target_link_libraries(coooda_cuda PUBLIC coooda_core CUDA::cudart coooda_options)
