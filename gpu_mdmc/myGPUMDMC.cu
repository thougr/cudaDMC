//
// Created by hougr t on 2024/1/28.
//

#include <chrono>
#include "myGPUMDMC.cuh"

#include "vtkArrayDispatch.h"
#include "vtkCellArray.h"
#include "vtkCharArray.h"
#include "vtkDataArrayRange.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkImageTransform.h"
#include "vtkIncrementalPointLocator.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkIntArray.h"
#include "vtkLongArray.h"
#include "vtkMarchingCubesTriangleCases.h"
#include "vtkMath.h"
#include "vtkMergePoints.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkShortArray.h"
#include "vtkStreamingDemandDrivenPipeline.h"


vtkStandardNewMacro(myGPUMDMC);
#define GLM_FORCE_CUDA
#include "../glm/glm.hpp"
#include "../table.cuh"
#include "../common.cuh"
#include "../Octree.cuh"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

__host__ __device__ glm::vec3 convertToRelative(Direction dir) {
    glm::vec3 v(0, 0, 0);
    switch (dir) {
        case Direction::BOTTOM:
            v = {0, 0, -1};
            break;
        case Direction::TOP:
            v = {0, 0, 1};
            break;
        case Direction::FRONT:
            v = {0, -1, 0};
            break;
        case Direction::BACK:
            v = {0, 1, 0};
            break;
        case Direction::LEFT:
            v = {-1, 0, 0};
            break;
        case Direction::RIGHT:
            v = {1, 0, 0};
            break;
        default:
            break;
    }
    return v;
}

template<class T, class B>
__global__ void generateLeafNodes(VoxelsData<T> *voxelsData, Octree<T,B> *leafNodes, int size, int depth, double isovalue, OctreeRepresentative *representatives) {
    unsigned stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//    unsigned blockId = (gridDim.x * gridDim.y * blockIdx.z) + (gridDim.x * blockIdx.y) + blockIdx.x;
//    unsigned offset = (blockId * (blockDim.x * blockDim.y * blockDim.z)) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for (int i = offset; i < size;i += stride) {
        leafNodes[i] = Octree<T,B>();
        auto &leaf = leafNodes[i];

        leaf.type = OctreeNodeType::Node_Leaf;
        leaf.height = 0;
        leaf.depth = depth;
        leaf.isoValue = isovalue;
        unsigned x = i % voxelsData->cubeDims.x;
        unsigned y = (i / voxelsData->cubeDims.x) % voxelsData->cubeDims.y;
        unsigned z = i / (voxelsData->cubeDims.x * voxelsData->cubeDims.y);
        unsigned index = (x % 2) + ((y % 2) << 1) + ((z % 2) << 2);
        glm::u32vec3 origin;
        origin.x = x;
        origin.y = y;
        origin.z = z;
        glm::u32vec3 regionSize;
        regionSize.x = 1;
        regionSize.y = 1;
        regionSize.z = 1;
        Region region = {.origin = origin, .size = regionSize, .voxelsCnt = 1, .conceptualSize = regionSize};
        leaf.region = region;
        leaf.index = index;
        auto p2Index = position2Index(region.origin, voxelsData->dims);
        leaf.maxScalar = voxelsData->scalars[p2Index];
        leaf.minScalar = leaf.maxScalar;
        for (int j = 0; j < 8; j++) {
            glm::u32vec3 verticesPos = region.origin + device_localVerticesPos[j];
//            verticesPos.x += localVerticesPos[j].x;
//            verticesPos.y += localVerticesPos[j].y;
//            verticesPos.z += localVerticesPos[j].z;
            auto scalar = voxelsData->scalars[position2Index(verticesPos, voxelsData->dims)];
            double n[3];
            vtkMarchingCubesComputePointGradient(verticesPos, voxelsData->scalars, voxelsData->dims, voxelsData->dims[0] * voxelsData->dims[1], n);
            leaf.normal[j] = {n[0], n[1], n[2]};
            if (scalar > leaf.maxScalar) {
                leaf.maxScalar = scalar;
            }
            if (scalar < leaf.minScalar) {
                leaf.minScalar = scalar;
            }
            leaf.scalar[j] = scalar;
            leaf.sign |= (scalar >= isovalue) ? 1 << j : 0;
        }

        for (int j = 0; j < 6; j++) {
            Direction dir = face_dir[j];
            glm::vec3 v = convertToRelative(dir);
            glm::vec3 neighbourPos = v + glm::vec3(origin);
//            neighbourPos.x += origin.x;
//            neighbourPos.y += origin.y;
//            neighbourPos.z += origin.z;
            if (neighbourPos.x >= voxelsData->cubeDims.x || neighbourPos.y >= voxelsData->cubeDims.y || neighbourPos.z >= voxelsData->cubeDims.z) {
                leaf.neighbour[j] = nullptr;
                continue;
            }
            if (neighbourPos.x < 0 || neighbourPos.y < 0 || neighbourPos.z < 0) {
                leaf.neighbour[j] = nullptr;
                continue;
            }
            leaf.neighbour[j] = &leafNodes[position2Index(neighbourPos, voxelsData->cubeDims)];
        }

        if (leaf.maxScalar < isovalue || leaf.minScalar > isovalue) {
            leafNodes[i].type = Node_None;
        } else {
            Octree<T,B>::calculateMDCRepresentative(&leaf, &leaf, representatives, i, isovalue, 1);
        }
    }
}

template<class T, class B>
__global__ void generateInternalNodes(VoxelsData<T> *voxelsData, Octree<T,B> *internalNodes, Octree<T,B> *childrenNodes,
                                      int size, int depth, int height, double isovalue,
                                      glm::u32vec3 childrenDims, glm::u32vec3 dims, unsigned regionSize) {
    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i = offset; i < size;i += stride){
        internalNodes[i] = Octree<T,B>();
        auto &leaf = internalNodes[i];
        leaf.depth = depth;
        leaf.height = height;
        unsigned x = i % dims.x;
        unsigned y = (i / dims.x) % dims.y;
        unsigned z = i / (dims.x * dims.y);
        unsigned index = (x % 2) + ((y % 2) << 1) + ((z % 2) << 2);
        glm::u32vec3 origin = {x, y, z};
        origin *= regionSize;
        glm::u32vec3 maxBound = origin + regionSize;
        maxBound = glm::min(maxBound, voxelsData->dims);
        maxBound -= origin;

        leaf.index = index;
        Region region = {.origin = origin , .size = maxBound, .voxelsCnt = maxBound.x * maxBound.y * maxBound.z, .conceptualSize = {regionSize, regionSize, regionSize}};
        leaf.region = region;
        leaf.type = OctreeNodeType::Node_Internal;
        leaf.isoValue = isovalue;

        T maxScalar = voxelsData->scalars[position2Index(region.origin, voxelsData->dims)];
        T minScalar = maxScalar;
        unsigned childrenSize = regionSize / 2u;
        for (int j = 0; j < 8; j++) {
            glm::u32vec3 relativeOrigin = origin + orderOrigin[j] * childrenSize;
            if (relativeOrigin.x >= voxelsData->cubeDims.x || relativeOrigin.y >= voxelsData->cubeDims.y || relativeOrigin.z >= voxelsData->cubeDims.z) {
                leaf.children[j] = nullptr;
                continue;
            }
            leaf.children[j] = &childrenNodes[position2Index(relativeOrigin / childrenSize, childrenDims)];
            if (leaf.children[j]->type == Node_None) {
                leaf.children[j] = nullptr;
                continue;
            }
            if (leaf.children[j]->maxScalar > maxScalar) {
                maxScalar = leaf.children[j]->maxScalar;
            }
            if (leaf.children[j]->minScalar < minScalar) {
                minScalar = leaf.children[j]->minScalar;
            }
        }
        leaf.maxScalar = maxScalar;
        leaf.minScalar = minScalar;
        if (leaf.maxScalar < isovalue || leaf.minScalar > isovalue) {
            leaf.type = Node_None;
        }
    }
}


template<class T, class B>
__global__ void calculateVerticesCnt(Octree<T,B> *leafNodes, int size, int *verticesCnt) {
    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for (int i = offset; i < size; i += stride) {
        auto leaf = &leafNodes[i];
        if (leaf->type == Node_None) {
            continue;
        }
        int curCnt = 0;
        OctreeRepresentative *vs[12];
        for (int j = 0; j < 12; j++) {
            auto representative = leaf->representative[j];
            if (representative == nullptr) {
                continue;
            }
            representative = findRepresentative(representative);
            bool isExist = false;
            for (int z = 0; z < curCnt; z++) {
                if (vs[z] == representative) {
                    isExist = true;
                    break;
                }
            }
            if (isExist) {
                continue;
            }
            vs[curCnt++] = representative;
            atomicAdd(verticesCnt, 1);
        }

    }
}


template<class T, class B>
__global__ void generateVerticesIndices(Octree<T,B> *leafNodes, int size, int *verticesIndices, glm::vec3 *vertices) {
    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for (int i = offset; i < size; i += stride) {
        auto leaf = &leafNodes[i];
        if (leaf->type == Node_None) {
            continue;
        }
        int curCnt = 0;
        OctreeRepresentative *vs[12];
        int indices[12];
        for (int j = 0; j < 12; j++) {
            auto representative = leaf->representative[j];
            if (representative == nullptr) {
                continue;
            }
            representative = findRepresentative(representative);
            bool isExist = false;
            for (int z = 0; z < curCnt; z++) {
                if (vs[z] == representative) {
                    isExist = true;
                    break;
                }
            }
            if (isExist) {
                continue;
            }
            int currIndex = atomicAdd(verticesIndices, 1);
            vs[curCnt] = representative;
            indices[curCnt] = currIndex;
            representative->index = currIndex;
            vertices[currIndex] = representative->position;
            curCnt++;
        }

    }
}
namespace {

    struct ComputeGradientWorker
    {
        template <class ScalarArrayT>
        __host__ void operator()(ScalarArrayT* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                        vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                        vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) const
        {
            int extent[6];
            double value = values[0];

            vtkInformation* inInfo = self->GetExecutive()->GetInputInformation(0, 0);
            inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent);
            using ComponentRef =  typename vtk::detail::SelectValueRange<ScalarArrayT, 1>::type;
            using ArrayType  = decltype(scalarsArray->GetArrayType());
            using OctreeType = Octree<ArrayType, ComponentRef>;
            glm::u32vec3 size =  {dims[0] - 1, dims[1] - 1, dims[2] - 1};
            int height = 0;
            glm::u32vec3 conceptualSize = findLargerClosestPower2Vector(size, height);
            auto numberOfTuples = scalarsArray->GetNumberOfTuples();
            auto scalars = vtk::DataArrayValueRange<1>(scalarsArray);

            VoxelsData<ArrayType> voxelsData = {.scalars = new ArrayType[numberOfTuples],
                    .cubeDims = size, .dims = {dims[0], dims[1], dims[2]}, .conceptualDims = conceptualSize };
            for (int i = 0; i < numberOfTuples; i++) {
                voxelsData.scalars[i] = scalars[i];
            }

            VoxelsData<ArrayType> *deviceData;
            cudaMallocManaged(&deviceData, sizeof(VoxelsData<ArrayType>));
            cudaMemcpy(deviceData, &voxelsData, sizeof(VoxelsData<ArrayType>), cudaMemcpyHostToDevice);
            ArrayType *deviceScalars;
            cudaMallocManaged(&deviceScalars, sizeof(ArrayType) * numberOfTuples);
            cudaMemcpy(deviceScalars, voxelsData.scalars, sizeof(ArrayType) * numberOfTuples, cudaMemcpyHostToDevice);
            // set deviceScalars to deviceData.scalars
            cudaMemcpy(&(deviceData->scalars), &deviceScalars, sizeof(ArrayType *), cudaMemcpyHostToDevice);
//            Region region = {.origin = {0, 0, 0}, .size = size,
//                    .voxelsCnt = size.x * size.y * size.z, .conceptualSize = conceptualSize};
            // open a file and read tree

            std::cout << voxelsData.scalars[0] << std::endl;
            std::cout << voxelsData.dims[0] << " " << voxelsData.dims[1] << " " << voxelsData.dims[2] << std::endl;
            std::cout << voxelsData.conceptualDims[0] << " " << voxelsData.conceptualDims[1] << " " << voxelsData.conceptualDims[2] << std::endl;
            std::cout << voxelsData.cubeDims[0] << " " << voxelsData.cubeDims[1] << " " << voxelsData.cubeDims[2] << std::endl;

            dim3 grid(32,32);
            dim3 block(32,32);
//    dim3 block={16, 8, 8};
//    dim3 grid={conceptualSize.x / block.x, conceptualSize.y / block.y, conceptualSize.z / block.z};
            // better grid and block for v100
            auto start = std::chrono::high_resolution_clock::now();;
            OctreeType *leafNodes = nullptr;
            OctreeRepresentative *octreeRepresentatives = nullptr;
            int leafNum = size.x * size.y * size.z;
            {
//                Region region = {.origin = {0, 0, 0}, .size = size,
//                        .voxelsCnt = size.x * size.y * size.z, .conceptualSize = conceptualSize};
                long long nByte = 1ll * sizeof(OctreeType) * leafNum;
                cudaMallocManaged((void**)&leafNodes, nByte);
                cudaMemset(leafNodes, 0, nByte);
                nByte = 4ll * sizeof(OctreeRepresentative) * leafNum;
                cudaMallocManaged((void**)&octreeRepresentatives, nByte);
                generateLeafNodes<<<grid, block>>>(deviceData, leafNodes, leafNum, height, value, octreeRepresentatives);
                auto error = cudaDeviceSynchronize();
                // print the error
                if (error != cudaSuccess) {
                    std::cout << cudaGetErrorString(error) << std::endl;
                }
            }

            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                double seconds = duration.count();
                start = end;
                std::cout << "构建叶子节点: " << seconds << " 秒" << std::endl;

//                int cnt = 0;
//                for (int i = 0; i < leafNum; i++) {
//                    auto &leaf = leafNodes[i];
////                    unsigned x = i % voxelsData.cubeDims.x;
////                    unsigned y = (i / voxelsData.cubeDims.x) % voxelsData.cubeDims.y;
////                    unsigned z = i / (voxelsData.cubeDims.x * voxelsData.cubeDims.y);
////            std::cout << x << " " << y << " " << z << std::endl;
////            std::cout << leaf.region.origin.x << " " << leaf.region.origin.y << " " << leaf.region.origin.z << std::endl;
////            std::cout << position2Index(leaf.region.origin, voxelsData.dims) << std::endl;
////            std::cout << leaf.maxScalar << " " << leaf.minScalar << std::endl;
////            std::cout << leaf.normal[0].x << " " << leaf.normal[0].y << " " << leaf.normal[0].z << std::endl;
//                    if (leaf.type == Node_None) {
//                        continue;
//                    }
////                    OctreeType::calculateMDCRepresentative(&leaf, &leaf, value, 1);
//                    cnt++;
//                }
//                std::cout << "leaf node num:" << cnt << std::endl;
//                return;
            }


            OctreeType **everyHeightNodes = new OctreeType*[height+1];
            everyHeightNodes[0] = leafNodes;
            unsigned regionSize = 1;
            glm::u32vec3 childrenSize = size;
            for (int h = 1; h <= height; h++) {
                OctreeType *nodes = nullptr;
                regionSize *= 2;
                // roundUp
                glm::u32vec3 curSize = (childrenSize + 1u) / 2u;
                int nodeNums = curSize.x * curSize.y * curSize.z;
                long long nByte = 1ll * sizeof(OctreeType) * nodeNums;
                cudaMallocManaged((void**)&nodes, nByte);
                cudaMemset(nodes, 0, nByte);
//                std::cout << nodeNums << std::endl;
                generateInternalNodes<<<grid, block>>>(deviceData, nodes, everyHeightNodes[h-1], nodeNums, height - h, h, value, childrenSize, curSize, regionSize);
                cudaDeviceSynchronize();
                everyHeightNodes[h] = nodes;
//                for (int i = 0; i < nodeNums; i++) {
//                    auto &node = nodes[i];
//                    if (node.type == Node_None) {
//                        continue;
//                    }
//                    1 == 1;
////            std::cout << node.region.origin.x << " " << node.region.origin.y << " " << node.region.origin.z << std::endl;
//                }
                childrenSize = curSize;
            }
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                double seconds = duration.count();
                start = end;
                std::cout << "构建中间节点: " << seconds << " 秒" << std::endl;
            }

            OctreeType *root = everyHeightNodes[height];


            OctreeType *isoTree = root;


// isoTree = OctreeType::simplifyOctree(isoTree, 1.1f);
//            OctreeType::clusterCell(isoTree, 1000.1f, 1);
            auto adaptiveThreshold = self->GetAdaptiveThreshold();
//            if (adaptiveThreshold != 0) {
//                OctreeType::clusterCell(isoTree, adaptiveThreshold, 1);
//            }
            OctreeType::generateVerticesIndices(isoTree, locator, newScalars);
//            int *d_count;
//            cudaMallocManaged(&d_count, sizeof(int ));
//            cudaMemset(d_count, 0, sizeof(int ));
//            calculateVerticesCnt<<<grid, block>>>(leafNodes, leafNum, d_count);
//            cudaDeviceSynchronize();
//            std::cout << "顶点个数:" << d_count[0] << std::endl;
//            glm::vec3 *vertices;
//            int *vertexIndex;
//            cudaMallocManaged(&vertexIndex, sizeof(int));
//            cudaMemset(vertexIndex, 0, sizeof(int));
//            cudaMallocManaged(&vertices, 1ll * sizeof(glm::vec3) * d_count[0]);
//            generateVerticesIndices<<<grid, block>>>(leafNodes, leafNum, vertexIndex, vertices);
//            cudaDeviceSynchronize();
//            std::cout << "数组顶点个数:" << vertexIndex[0] << std::endl;
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                double seconds = duration.count();
                start = end;
                std::cout << "生成顶点: " << seconds << " 秒" << std::endl;
            }
            OctreeType::contourCellProc(isoTree, newPolys, 1);


//            OctreeType::destroyOctree(isoTree);


//            Octree<ArrayType,ComponentRef>::destroyOctree(allDataTree);
//            Octree<ScalarArrayT>::buildOctree()
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                double seconds = duration.count();
                start = end;
                std::cout << "生成三角面: " << seconds << " 秒" << std::endl;
            }


            cudaFree(deviceData);
            delete voxelsData.scalars;
            // free the everyHeightNodes
            for (int i = 0; i <= height; i++) {
                cudaFree(everyHeightNodes[i]);
            }
            delete []everyHeightNodes;
//            cudaFree(d_count);
//            cudaFree(vertices);
//            cudaFree(vertexIndex);
        }
    };



}

void myGPUMDMC::process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                           vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                           vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) {

    using Dispatcher = vtkArrayDispatch::Dispatch;
    ComputeGradientWorker worker;
    if (!Dispatcher::Execute(scalarsArray, worker, this, dims, this->Locator, newScalars, newGradients,
                             newNormals, newPolys, values, numValues)) { // Fallback to slow path for unknown arrays:
        std::cout << "Fallback to slow path for unknown arrays" << std::endl;
        worker(scalarsArray, this, dims, this->Locator, newScalars, newGradients, newNormals, newPolys,
               values, numValues);
    }
}

bool myGPUMDMC::supportAdaptiveMeshing() {
    return true;
}
