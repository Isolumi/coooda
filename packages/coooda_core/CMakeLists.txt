add_library(coooda_core
  src/benchmark.cpp
  src/cuda_check.cu
  src/device.cu
  src/status.cpp
  src/tensor.cpp
  src/test_harness.cpp
)

target_include_directories(coooda_core PUBLIC include)
target_link_libraries(coooda_core PUBLIC CUDA::cudart coooda_options)
