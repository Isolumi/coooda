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

add_library(coooda_options INTERFACE)
target_compile_options(coooda_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-lineinfo>
)

add_subdirectory(packages/coooda_core)
add_subdirectory(packages/coooda_cpp)
add_subdirectory(packages/coooda_cuda)
add_subdirectory(packages/coooda_compare)

foreach(coooda_optional_dir tests bench apps)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${coooda_optional_dir}/CMakeLists.txt")
    add_subdirectory(${coooda_optional_dir})
  endif()
endforeach()
