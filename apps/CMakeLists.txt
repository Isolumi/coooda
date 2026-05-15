add_executable(inspect_device inspect_device.cu)
target_link_libraries(inspect_device PRIVATE coooda_core)

add_executable(run_checkpoint run_checkpoint.cpp)
target_link_libraries(run_checkpoint PRIVATE coooda_core coooda_cpp)

add_executable(compare_backends compare_backends.cu)
target_link_libraries(compare_backends PRIVATE coooda_core coooda_compare)

add_executable(tiny_gpt_cpp tiny_gpt_cpp.cpp)
target_link_libraries(tiny_gpt_cpp PRIVATE coooda_core coooda_cpp)

add_executable(tiny_gpt_cuda tiny_gpt_cuda.cu)
target_link_libraries(tiny_gpt_cuda PRIVATE coooda_core coooda_cuda)

foreach(coooda_app_target
    inspect_device
    run_checkpoint
    compare_backends
    tiny_gpt_cpp
    tiny_gpt_cuda
)
    set_target_properties(${coooda_app_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps"
    )
endforeach()
