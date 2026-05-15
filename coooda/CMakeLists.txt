cmake_minimum_required(VERSION 3.24)

project(coooda LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_library(coooda
  src/bench/benchmark.cpp
  src/core/cuda_checks.cu
  src/core/device.cu
  src/core/timers.cu
  src/tests/validation.cpp
)

target_include_directories(coooda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(coooda
  PUBLIC
    CUDA::cudart
)

target_compile_definitions(coooda
  PUBLIC
    COODA_BUILD_MODE="$<CONFIG>"
)

target_compile_options(coooda
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-lineinfo>
)

add_executable(coooda_tests
  tests/coooda_tests.cpp
)

target_link_libraries(coooda_tests
  PRIVATE
    coooda
)

add_test(NAME coooda_tests COMMAND coooda_tests)

add_executable(coooda_bench
  bench/coooda_bench.cu
)

target_link_libraries(coooda_bench
  PRIVATE
    coooda
)

add_executable(coooda_tiny_gpt
  apps/tiny_gpt.cpp
)

target_link_libraries(coooda_tiny_gpt
  PRIVATE
    coooda
)
