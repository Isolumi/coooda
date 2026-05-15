add_library(coooda_compare
  src/ops/vector_compare.cu
  src/package.cpp
)

target_include_directories(coooda_compare PUBLIC include)
target_link_libraries(coooda_compare PUBLIC coooda_core coooda_cpp coooda_cuda coooda_options)
