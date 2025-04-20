cmake_minimum_required(VERSION 3.18)
project(drl_model)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create DLL
add_library(DRLModel SHARED
    src/model_wrapper.cpp
    src/dll_interface.cpp
)

target_include_directories(DRLModel PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(DRLModel PRIVATE
    ${TORCH_LIBRARIES}
)

# Set output directories
set_target_properties(DRLModel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Copy DLL to output directory
add_custom_command(TARGET DRLModel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:DRLModel>
        ${CMAKE_BINARY_DIR}/bin/DRLModel.dll
)
