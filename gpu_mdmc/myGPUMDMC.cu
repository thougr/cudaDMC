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
__host__ __device__ Octree<T,B> * getNeighbour(VoxelsData<T> *voxelsData, Octree<T,B> *leafNodes, glm::vec3 v, glm::u32vec3 origin) {
    glm::vec3 neighbourPos = v + glm::vec3(origin);
    if (neighbourPos.x >= voxelsData->cubeDims.x || neighbourPos.y >= voxelsData->cubeDims.y || neighbourPos.z >= voxelsData->cubeDims.z) {
        return nullptr;
    }
    if (neighbourPos.x < 0 || neighbourPos.y < 0 || neighbourPos.z < 0) {
        return nullptr;
    }
    return &leafNodes[position2Index(neighbourPos, voxelsData->cubeDims)];
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
            Direction dir = static_cast<Direction>(j);
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
template<class T, class B>
__host__ __device__ bool isAllNodeValid(const Octree<T,B> *nodes[4]) {
    for (int i = 0; i < 4; i++) {
        if (nodes[i] == nullptr) {
            return false;
        }
        if (nodes[i]->type == Node_None) {
            return false;
        }
    }
    return true;
}

template<class T, class B>
__device__ void contourProcessEdge(const Octree<T, B> *root[4], int dir, glm::u32vec3 *newPolys, int triIndex, int useOptimization) {
    int minIndex = 0;
    long long indices[4] = {-1, -1, -1, -1};
    bool flip = false;
    bool signChange[4] = {false, false, false, false};
    OctreeRepresentative *representatives[4];
    int intersections[4] = {0, 0, 0, 0};
    auto minAngle = [&] (const glm::vec3 v0, const glm::vec3 v1, const glm::vec3 v2) -> double
    {
        const float da = glm::distance(v0, v1); // std::sqrt((v1.x - v0.x) * (v1.x - v0.x) + (v1.y - v0.y) * (v1.y - v0.y) + (v1.z - v0.z) * (v1.z - v0.z));
        const float db = glm::distance(v1, v2); // std::sqrt((v2.x - v1.x) * (v2.x - v1.x) + (v2.y - v1.y) * (v2.y - v1.y) + (v2.z - v1.z) * (v2.z - v1.z));
        const float dc = glm::distance(v2, v0); // std::sqrt((v0.x - v2.x) * (v0.x - v2.x) + (v0.y - v2.y) * (v0.y - v2.y) + (v0.z - v2.z) * (v0.z - v2.z));
        const float dA = std::acos((db * db + dc * dc - da * da) / (2 * db * dc));
        const float dB = std::acos((da * da + dc * dc - db * db) / (2 * da * dc));
        const float dC = std::acos((db * db + da * da - dc * dc) / (2 * db * da));

        return min(min(dA, dB), dC);
    };
    for (int i = 0; i < 4; i++) {
        const int edge = device_processEdgeMask[dir][i];
        int c1 = device_edges2Vertices[edge][0];
        int c2 = device_edges2Vertices[edge][1];
        auto m1 = (root[i]->sign >> c1) & 1;
        auto m2 = (root[i]->sign >> c2) & 1;
        auto vertex = root[i]->representative[edge];

        if (m1 ^ m2) {
            signChange[i] = true;
            vertex = findRepresentative(vertex);
            representatives[i] = vertex;
            indices[i] = vertex->index;
            intersections[i] = vertex->edgeIntersection[edge];
        }
        flip = m1;
    }
    auto isAllDifferent = [&] (long long *tris, int len) {
        // more quickly, use unordered_set
//        std::unordered_set<long long> set;
        for (int i = 0; i < 3; i++) {
            if (tris[i] == -1) {
                return false;
            }
        }
        return tris[0] != tris[1] && tris[0] != tris[2] && tris[1] != tris[2];
//        for (int i = 0; i < len; i++) {
//            if (set.find(tris[i]) != set.end() || tris[i] == -1) {
//                return false;
//            }
//            set.insert(tris[i]);
//        }
//        return true;
    };

    auto insertTriangle = [&] (long long verticesId[4], const int indices[4], glm::u32vec3 *newPolys, int index) {
        int tris1[3] = {0, 2, 3};
        int tris2[3] = {0, 1, 2};
        if (intersections[indices[0]] == 2 && intersections[indices[2]] == 2) {
            tris1[1] = 1;
            tris2[0] = 1;
            tris2[1] = 2;
            tris2[2] = 3;
//            tris1 = {0, 1, 3};
//            tris2 = {1, 2, 3};
        }
        {
            long long tris[3];
            int realIndices[3];
            for (int i = 0; i < 3; i++) {
                auto realIndex = indices[tris1[i]];
                realIndices[i] = realIndex;
                tris[i] = verticesId[realIndex];
            }
            if (isAllDifferent(tris, 3)) {
                newPolys[index] = glm::u32vec3(tris[0], tris[1], tris[2]);
            }
        }
        {
            long long tris[3];
            int realIndices[3];
            for (int i = 0; i < 3; i++) {
                auto realIndex = indices[tris2[i]];
                realIndices[i] = realIndex;
                tris[i] = verticesId[realIndex];
            }
            if (isAllDifferent(tris, 3)) {
                newPolys[index + 1] = glm::u32vec3(tris[0], tris[1], tris[2]);
            }
        }
    };

    if (signChange[minIndex]) {
        auto v0 = representatives[0]->position;
        auto v1 = representatives[1]->position;
        auto v2 = representatives[2]->position;
        auto v3 = representatives[3]->position;
        double a1_ = minAngle(v0, v1, v2);
        double a2_ = minAngle(v2, v3, v1);
        const double b1_ = min(a1_, a2_);
        const double b2_ = max(a1_, a2_);
        a1_ = minAngle(v0, v1, v3);
        a2_ = minAngle(v0, v2, v3);
        const double c1_ = min(a1_, a2_);
        const double c2_ = max(a1_, a2_);
//        if (flip) {
//            const int ins[4] = {0, 2, 3, 1};
//            insertTriangle(indices, ins, newPolys);
//
//        } else {
//            const int ins[4] = {0, 1, 3, 2};
//            insertTriangle(indices, ins, newPolys);
//        }

        if (!useOptimization || b1_ < c1_ || (b1_ == c1_ && b2_ <= c2_))
        {
            if (flip) {
                const int ins[4] = {0, 2, 3, 1};
                insertTriangle(indices, ins, newPolys, triIndex);

            } else {
                const int ins[4] = {0, 1, 3, 2};
                insertTriangle(indices, ins, newPolys, triIndex);
            }
        } else {
            if (flip) {
                const int ins[4] = {2, 3, 1, 0};
                insertTriangle(indices, ins, newPolys, triIndex);

            } else {
                const int ins[4] = {2, 0, 1, 3};
                insertTriangle(indices, ins, newPolys, triIndex);
            }
        }
    }
}

template<class T, class B>
__global__ void calculateQuadCnt(VoxelsData<T> *voxelsData, Octree<T,B> *leafNodes, int size, int *quadCnt) {
    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for (int i = offset; i < size; i += stride) {
        auto leaf = &leafNodes[i];
        if (leaf->type == Node_None) {
            continue;
        }
        const int edgeCheck[3] = {5, 6, 10};
        const int dirs[3] = {1, 0, 2};
        if (leaf->representative[5] != nullptr) {
            const Octree<T,B> *neighbourCheck[4] = {leaf,
                                                    leaf->neighbour[Direction::RIGHT],
                                                    leaf->neighbour[Direction::TOP],
//            getNeighbour(voxelsData, leafNodes, {1, 0, 0}, leaf->region.origin),
//            getNeighbour(voxelsData, leafNodes, {0, 0, 1}, leaf->region.origin),
                                                    getNeighbour(voxelsData, leafNodes, {1, 0, 1}, leaf->region.origin)};
            if (isAllNodeValid(neighbourCheck)) {
                atomicAdd(quadCnt, 1);
            }
        }

        if (leaf->representative[6] != nullptr) {
            const Octree<T,B> *neighbourCheck[4] = {leaf,
                                                    leaf->neighbour[Direction::TOP],
                                                    leaf->neighbour[Direction::BACK],
//                                                    getNeighbour(voxelsData, leafNodes, {0, 0, 1}, leaf->region.origin),
//                                                    getNeighbour(voxelsData, leafNodes, {0, 1, 0}, leaf->region.origin),
                                                    getNeighbour(voxelsData, leafNodes, {0, 1, 1}, leaf->region.origin)};
            if (isAllNodeValid(neighbourCheck)) {
                atomicAdd(quadCnt, 1);
            }
        }

        if (leaf->representative[10] != nullptr) {
            const Octree<T,B> *neighbourCheck[4] = {leaf,
                                                    leaf->neighbour[Direction::BACK],
                                                    leaf->neighbour[Direction::RIGHT],

//                                                    getNeighbour(voxelsData, leafNodes, {0, 1, 0}, leaf->region.origin),
//                                                    getNeighbour(voxelsData, leafNodes, {1, 0, 0}, leaf->region.origin),

                                                    getNeighbour(voxelsData, leafNodes, {1, 1, 0}, leaf->region.origin)};
            if (isAllNodeValid(neighbourCheck)) {
                atomicAdd(quadCnt, 1);
            }
        }
    }
}



template<class T, class B>
__global__ void generateQuad(VoxelsData<T> *voxelsData, Octree<T,B> *leafNodes, int size, glm::u32vec3 *tris, int *triIndex) {
    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for (int i = offset; i < size; i += stride) {
        auto leaf = &leafNodes[i];
        if (leaf->type == Node_None) {
            continue;
        }
        const int edgeCheck[3] = {5, 6, 10};
        const int dirs[3] = {1, 0, 2};
        if (leaf->representative[5] != nullptr) {
            const Octree<T,B> *neighbourCheck[4] = {leaf,
                                                    leaf->neighbour[Direction::RIGHT],
                                                    leaf->neighbour[Direction::TOP],
//            getNeighbour(voxelsData, leafNodes, {1, 0, 0}, leaf->region.origin),
//            getNeighbour(voxelsData, leafNodes, {0, 0, 1}, leaf->region.origin),
                                                    getNeighbour(voxelsData, leafNodes, {1, 0, 1}, leaf->region.origin)};
            if (isAllNodeValid(neighbourCheck)) {
                auto index = atomicAdd(triIndex, 2);
                contourProcessEdge(neighbourCheck, dirs[0], tris, index, 1);
            }
        }

        if (leaf->representative[6] != nullptr) {
            const Octree<T,B> *neighbourCheck[4] = {leaf,
                                                    leaf->neighbour[Direction::TOP],
                                                    leaf->neighbour[Direction::BACK],
//                                                    getNeighbour(voxelsData, leafNodes, {0, 0, 1}, leaf->region.origin),
//                                                    getNeighbour(voxelsData, leafNodes, {0, 1, 0}, leaf->region.origin),
                                                    getNeighbour(voxelsData, leafNodes, {0, 1, 1}, leaf->region.origin)};
            if (isAllNodeValid(neighbourCheck)) {
                auto index = atomicAdd(triIndex, 2);
                contourProcessEdge(neighbourCheck, dirs[1], tris, index, 1);
            }
        }

        if (leaf->representative[10] != nullptr) {
            const Octree<T,B> *neighbourCheck[4] = {leaf,
                                                    leaf->neighbour[Direction::BACK],
                                                    leaf->neighbour[Direction::RIGHT],

//                                                    getNeighbour(voxelsData, leafNodes, {0, 1, 0}, leaf->region.origin),
//                                                    getNeighbour(voxelsData, leafNodes, {1, 0, 0}, leaf->region.origin),

                                                    getNeighbour(voxelsData, leafNodes, {1, 1, 0}, leaf->region.origin)};
            if (isAllNodeValid(neighbourCheck)) {
                auto index = atomicAdd(triIndex, 2);
                contourProcessEdge(neighbourCheck, dirs[2], tris, index, 1);
            }
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
            auto begin = std::chrono::high_resolution_clock::now();;
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
                generateInternalNodes<<<grid, block>>>(deviceData, nodes, everyHeightNodes[h-1], nodeNums, height - h, h, value, childrenSize, curSize, regionSize);
                cudaDeviceSynchronize();
                everyHeightNodes[h] = nodes;
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
//            OctreeType::generateVerticesIndices(isoTree, locator, newScalars);
            int *d_count;
            cudaMallocManaged(&d_count, sizeof(int ));
            cudaMemset(d_count, 0, sizeof(int ));
            calculateVerticesCnt<<<grid, block>>>(leafNodes, leafNum, d_count);
            cudaDeviceSynchronize();
            std::cout << "顶点个数:" << d_count[0] << std::endl;
            glm::vec3 *vertices;
            int *vertexIndex;
            cudaMallocManaged(&vertexIndex, sizeof(int));
            cudaMemset(vertexIndex, 0, sizeof(int));
            cudaMallocManaged(&vertices, 1ll * sizeof(glm::vec3) * d_count[0]);
            generateVerticesIndices<<<grid, block>>>(leafNodes, leafNum, vertexIndex, vertices);
            cudaDeviceSynchronize();
            std::cout << "数组顶点个数:" << vertexIndex[0] << std::endl;
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                double seconds = duration.count();
                start = end;
                std::cout << "生成顶点: " << seconds << " 秒" << std::endl;
            }


            int *quadCnt;
            cudaMallocManaged(&quadCnt, sizeof(int));
            cudaMemset(quadCnt, 0, sizeof(int));
            calculateQuadCnt<<<grid, block>>>(deviceData, leafNodes, leafNum, quadCnt);
            cudaDeviceSynchronize();
            std::cout << "三角面个数:" << quadCnt[0] * 2 << std::endl;
            glm::u32vec3 *tris;
            int *trisIndex;
            cudaMallocManaged(&trisIndex, sizeof(int));
            cudaMemset(trisIndex, 0, sizeof(int));
            cudaMallocManaged(&tris, 2ll * sizeof(glm::u32vec3) * quadCnt[0]);
            generateQuad<<<grid, block>>>(deviceData, leafNodes, leafNum, tris, trisIndex);
            cudaDeviceSynchronize();
            std::cout << "数组三角面个数:" << trisIndex[0] << std::endl;


//            OctreeType::contourCellProc(isoTree, newPolys, 1);


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
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - begin;
                double seconds = duration.count();
                std::cout << "总时间: " << seconds << " 秒" << std::endl;
            }

            std::unordered_map<int, vtkIdType > vertexMap;
            for (int i = 0; i < vertexIndex[0]; i++) {
                vtkIdType id;
                double p[3];
                for (int j = 0; j < 3; j++) {
                    p[j] = vertices[i][j];
                }
                if (locator->InsertUniquePoint(p, id)) {
                    newScalars->InsertTuple(id, &value);
                }
                vertexMap[i] = id;
            }
            for (int i = 0; i < trisIndex[0]; i++) {
                vtkIdType ids[3];
                for (int j = 0; j < 3; j++) {
                    ids[j] = vertexMap[tris[i][j]];
                }
                newPolys->InsertNextCell(3, ids);
            }

            cudaFree(deviceData);
            delete voxelsData.scalars;
            // free the everyHeightNodes
            for (int i = 0; i <= height; i++) {
                cudaFree(everyHeightNodes[i]);
            }
            delete []everyHeightNodes;
            cudaFree(d_count);
            cudaFree(vertices);
            cudaFree(vertexIndex);
            cudaFree(quadCnt);
            cudaFree(tris);
            cudaFree(trisIndex);
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
