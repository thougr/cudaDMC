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
        UnionFind.h
        cuda_tmc/MarchingCubes.cu
        cuda_tmc/vtk/gpuMC.cu
        cuda_tmc/vtk/gpuMC.cuh
        dmc/vtk/gpuDMC.cu
        dmc/vtk/gpuDMC.cuh
        dmc/CellIntersection.h
        dmc/CTimer.h
        dmc/DualMarchingCubes.cu
        dmc/EdgeHashTable.h
        dmc/Edges.h
        dmc/ElementQuality.h
        dmc/EstimateElementQuality.cu
        dmc/EstimateElementQuality.h
        dmc/FaceColoring.cu
        dmc/FaceColoring.h
        dmc/HalfedgeFaces.h
        dmc/HalfedgeMesh.cu
        dmc/HalfedgeHashTable.h
        dmc/HalfedgeMesh.h
        dmc/Halfedges.h
        dmc/HalfedgeVertices.h
        dmc/kernel.cu
        dmc/MarchingCubesLookupTables.h
        dmc/MCPolygon.h
        dmc/MeshSimplification.cu
        dmc/MeshSimplification.h
        dmc/QuadrilateralMap.h
        dmc/QuadrilateralHashTable.h
        dmc/Quadrilaterals.h
        dmc/QualityMeasure.h
        dmc/Triangles.h
        dmc/UniformGrid.cu
        dmc/ValenceHashTable.h
        dmc/VertexHashTable.h
        dmc/VertexMap.h
        dmc/VertexRepresentative.h
        dmc/VertexValence.cu
        dmc/VertexValence.h
        dmc/Vertices.h
)
target_compile_options(cudaDMC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_dependencies(cudaDMC copy_data)
set_target_properties(cudaDMC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        COMPILE_OPTIONS -G
)
target_link_libraries(cudaDMC ${VTK_LIBRARIES} ${NLOPT_LIBRARIES})