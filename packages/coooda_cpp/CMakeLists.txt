add_library(coooda_cpp
  src/ops/matmul.cpp
  src/ops/reductions.cpp
  src/ops/vector.cpp
  src/package.cpp
)

target_include_directories(coooda_cpp PUBLIC include)
target_link_libraries(coooda_cpp PUBLIC coooda_core coooda_options)
