cmake_minimum_required(VERSION 3.20)

# Project Name
project(STFGNN LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Toolkit
find_package(CUDA REQUIRED)
set(CUDA_ARCHITECTURES 75)

# Include Directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/eigen
)

# Source Files
set(SOURCES
    src/dataloader.cpp
    src/preprocessing.cu
    src/graph_construction.cpp
    src/stfgnn_layer.cu
    src/gated_conv.cu
    src/fusion_module.cpp
    src/level1_layers.cpp
    src/level0_model.cpp
    src/main.cpp
)

# Add Executable
add_executable(stfgnn ${SOURCES})

# Link CUDA
target_link_libraries(stfgnn PRIVATE CUDA::cudart)

# Compilation Flags
set_target_properties(stfgnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Output Settings
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
