cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.2/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2")
project(cudaDMC LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(VTK REQUIRED)
find_package(NLopt REQUIRED)
include(FetchContent)
FetchContent_Declare(
        stdgpu
        GIT_REPOSITORY https://github.com/stotko/stdgpu.git
        GIT_TAG        master
)

# Exclude unneeded parts from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(stdgpu)

    # print version
message(STATUS "CUDA version: ${CUDA_VERSION}")

    # 添加-G到CUDA NVCC编译器标志
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
    set(CUDA_NVCC_FLAGS "-g -G")
include_directories(${NLOPT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
# 设置源目录和目标目录变量
set(SOURCE_DATA_DIR "${CMAKE_SOURCE_DIR}/data" "${CMAKE_SOURCE_DIR}/large_data")
set(DEST_DATA_DIR "${CMAKE_BINARY_DIR}/data")

# 创建一个自定义目标，用于复制data目录
add_custom_target(copy_data ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Copying data directory to build directory..."
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${SOURCE_DATA_DIR} ${DEST_DATA_DIR}
        COMMENT "Copied data directory to build directory."
        )

add_executable(cudaDMC main.cu Octree.cuh qef.cu qef.cuh svd.cu svd.cuh Octree.inl table.cuh common.cuh
        isosurfacesAlgorithm.cpp
        isosurfacesAlgorithm.h
        gpu_mdmc/myGPUMDMC.cu
        gpu_mdmc/myGPUMDMC.cuh
        table.cu
        UnionFind.cuh
        UnionFindGPU.cuh)
target_compile_options(cudaDMC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_dependencies(cudaDMC copy_data)
set_target_properties(cudaDMC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        COMPILE_OPTIONS -G
)
target_link_libraries(cudaDMC ${VTK_LIBRARIES} ${NLOPT_LIBRARIES} stdgpu::stdgpu)