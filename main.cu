#include <iostream>

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "Octree.cuh"
#include <vtkNrrdReader.h>
#include <vtkNew.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <chrono>
#include <vtkOBJWriter.h>
#include "vtkDataArrayRange.h"
#include "common.cuh"
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "gpu_mdmc/myGPUMDMC.cuh"

#include <filesystem>

//template <class T>
//__device__ void vtkMarchingCubesComputePointGradient(
//        glm::u32vec3 pos, const T &s, glm::u32vec3 dims, int sliceSize, double n[3])
//{
//    auto i = pos[0], j = pos[1], k = pos[2];
//    double sp, sm;
//    // x-direction
//    if (i == 0)
//    {
//        sp = s[i + 1 + j * dims[0] + k * sliceSize];
//        sm = s[i + j * dims[0] + k * sliceSize];
//        n[0] = sm - sp;
//    }
//    else if (i == (dims[0] - 1))
//    {
//        sp = s[i + j * dims[0] + k * sliceSize];
//        sm = s[i - 1 + j * dims[0] + k * sliceSize];
//        n[0] = sm - sp;
//    }
//    else
//    {
//        sp = s[i + 1 + j * dims[0] + k * sliceSize];
//        sm = s[i - 1 + j * dims[0] + k * sliceSize];
//        n[0] = 0.5 * (sm - sp);
//    }
//
//    // y-direction
//    if (j == 0)
//    {
//        sp = s[i + (j + 1) * dims[0] + k * sliceSize];
//        sm = s[i + j * dims[0] + k * sliceSize];
//        n[1] = sm - sp;
//    }
//    else if (j == (dims[1] - 1))
//    {
//        sp = s[i + j * dims[0] + k * sliceSize];
//        sm = s[i + (j - 1) * dims[0] + k * sliceSize];
//        n[1] = sm - sp;
//    }
//    else
//    {
//        sp = s[i + (j + 1) * dims[0] + k * sliceSize];
//        sm = s[i + (j - 1) * dims[0] + k * sliceSize];
//        n[1] = 0.5 * (sm - sp);
//    }
//
//    // z-direction
//    if (k == 0)
//    {
//        sp = s[i + j * dims[0] + (k + 1) * sliceSize];
//        sm = s[i + j * dims[0] + k * sliceSize];
//        n[2] = sm - sp;
//    }
//    else if (k == (dims[2] - 1))
//    {
//        sp = s[i + j * dims[0] + k * sliceSize];
//        sm = s[i + j * dims[0] + (k - 1) * sliceSize];
//        n[2] = sm - sp;
//    }
//    else
//    {
//        sp = s[i + j * dims[0] + (k + 1) * sliceSize];
//        sm = s[i + j * dims[0] + (k - 1) * sliceSize];
//        n[2] = 0.5 * (sm - sp);
//    }
//};
//template class Octree<int, ::vtk::detail::ValueRange<::vtkDataArray, (int)1>>;
//template struct VoxelsData<int>;
//template void vtkMarchingCubesComputePointGradient<int*>(
//        glm::u32vec3 pos, int* const &s, glm::u32vec3 dims, int sliceSize, double n[3]);


//__host__ __device__ glm::vec3 convertToRelative(Direction dir) {
//    glm::vec3 v(0, 0, 0);
//    switch (dir) {
//        case Direction::BOTTOM:
//            v = {0, 0, -1};
//            break;
//        case Direction::TOP:
//            v = {0, 0, 1};
//            break;
//        case Direction::FRONT:
//            v = {0, -1, 0};
//            break;
//        case Direction::BACK:
//            v = {0, 1, 0};
//            break;
//        case Direction::LEFT:
//            v = {-1, 0, 0};
//            break;
//        case Direction::RIGHT:
//            v = {1, 0, 0};
//            break;
//        default:
//            break;
//    }
//    return v;
//}
//
//template<class T, class B>
//__global__ void generateLeafNodes(VoxelsData<T> *voxelsData, Octree<T,B> *leafNodes, int size, int depth, double isovalue) {
//    unsigned stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
//    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//    for (int i = offset; i < size;i += stride) {
//        leafNodes[i] = Octree<T,B>();
//        auto &leaf = leafNodes[i];
//
//        leaf.type = OctreeNodeType::Node_Leaf;
//        leaf.height = 0;
//        leaf.depth = depth;
//        leaf.isoValue = isovalue;
//        unsigned x = i % voxelsData->cubeDims.x;
//        unsigned y = (i / voxelsData->cubeDims.x) % voxelsData->cubeDims.y;
//        unsigned z = i / (voxelsData->cubeDims.x * voxelsData->cubeDims.y);
//        unsigned index = (x % 2) + ((y % 2) << 1) + ((z % 2) << 2);
//        glm::u32vec3 origin;
//        origin.x = x;
//        origin.y = y;
//        origin.z = z;
//        glm::u32vec3 regionSize;
//        regionSize.x = 1;
//        regionSize.y = 1;
//        regionSize.z = 1;
//        Region region = {.origin = origin, .size = regionSize, .voxelsCnt = 1, .conceptualSize = regionSize};
//        leaf.region = region;
//        leaf.index = index;
//        auto p2Index = position2Index(region.origin, voxelsData->dims);
//        leaf.maxScalar = voxelsData->scalars[p2Index];
//        leaf.minScalar = leaf.maxScalar;
//        for (int j = 0; j < 8; j++) {
//            glm::u32vec3 verticesPos = region.origin + localVerticesPos[j];
////            verticesPos.x += localVerticesPos[j].x;
////            verticesPos.y += localVerticesPos[j].y;
////            verticesPos.z += localVerticesPos[j].z;
//            auto scalar = voxelsData->scalars[position2Index(verticesPos, voxelsData->dims)];
//            double n[3];
//            vtkMarchingCubesComputePointGradient(verticesPos, voxelsData->scalars, voxelsData->dims, voxelsData->dims[0] * voxelsData->dims[1], n);
//            leaf.normal[j] = {n[0], n[1], n[2]};
//            if (scalar > leaf.maxScalar) {
//                leaf.maxScalar = scalar;
//            }
//            if (scalar < leaf.minScalar) {
//                leaf.minScalar = scalar;
//            }
//            leaf.scalar[j] = scalar;
//            leaf.sign |= (scalar >= isovalue) ? 1 << j : 0;
//        }
//
//        for (int j = 0; j < 6; j++) {
//            Direction dir = face_dir[j];
//            glm::vec3 v = convertToRelative(dir);
//            glm::vec3 neighbourPos = v + glm::vec3(origin);
////            neighbourPos.x += origin.x;
////            neighbourPos.y += origin.y;
////            neighbourPos.z += origin.z;
//            if (neighbourPos.x >= voxelsData->cubeDims.x || neighbourPos.y >= voxelsData->cubeDims.y || neighbourPos.z >= voxelsData->cubeDims.z) {
//                leaf.neighbour[j] = nullptr;
//                continue;
//            }
//            if (neighbourPos.x < 0 || neighbourPos.y < 0 || neighbourPos.z < 0) {
//                leaf.neighbour[j] = nullptr;
//                continue;
//            }
//            leaf.neighbour[j] = &leafNodes[position2Index(neighbourPos, voxelsData->cubeDims)];
//        }
//
//        if (leaf.maxScalar < isovalue || leaf.minScalar > isovalue) {
//            leafNodes[i].type = Node_None;
//        } else {
////            Octree<T,B>::calculateMDCRepresentative(&leaf, &leaf, isovalue, 1);
//        }
//    }
//}
//
//template<class T, class B>
//__global__ void generateInternalNodes(VoxelsData<T> *voxelsData, Octree<T,B> *internalNodes, Octree<T,B> *childrenNodes,
//                                      int size, int depth, int height, double isovalue,
//                                      glm::u32vec3 childrenDims, glm::u32vec3 dims, unsigned regionSize) {
//    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
//    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//    for(int i = offset; i < size;i += stride){
//        internalNodes[i] = Octree<T,B>();
//        auto &leaf = internalNodes[i];
//        leaf.depth = depth;
//        leaf.height = height;
//        unsigned x = i % dims.x;
//        unsigned y = (i / dims.x) % dims.y;
//        unsigned z = i / (dims.x * dims.y);
//        unsigned index = (x % 2) + ((y % 2) << 1) + ((z % 2) << 2);
//        glm::u32vec3 origin = {x, y, z};
//        origin *= regionSize;
//        glm::u32vec3 maxBound = origin + regionSize;
//        maxBound = glm::min(maxBound, voxelsData->dims);
//        maxBound -= origin;
//
//        leaf.index = index;
//        Region region = {.origin = origin , .size = maxBound, .voxelsCnt = maxBound.x * maxBound.y * maxBound.z, .conceptualSize = {regionSize, regionSize, regionSize}};
//        leaf.region = region;
//        leaf.type = OctreeNodeType::Node_Internal;
//        leaf.isoValue = isovalue;
//
//        T maxScalar = voxelsData->scalars[position2Index(region.origin, voxelsData->dims)];
//        T minScalar = maxScalar;
//        unsigned childrenSize = regionSize / 2u;
//        for (int j = 0; j < 8; j++) {
//            glm::u32vec3 relativeOrigin = origin + orderOrigin[j] * childrenSize;
//            if (relativeOrigin.x >= voxelsData->cubeDims.x || relativeOrigin.y >= voxelsData->cubeDims.y || relativeOrigin.z >= voxelsData->cubeDims.z) {
//                leaf.children[j] = nullptr;
//                continue;
//            }
//            leaf.children[j] = &childrenNodes[position2Index(relativeOrigin / childrenSize, childrenDims)];
//            if (leaf.children[j]->type == Node_None) {
//                leaf.children[j] = nullptr;
//                continue;
//            }
//            if (leaf.children[j]->maxScalar > maxScalar) {
//                maxScalar = leaf.children[j]->maxScalar;
//            }
//            if (leaf.children[j]->minScalar < minScalar) {
//                minScalar = leaf.children[j]->minScalar;
//            }
//        }
//        leaf.maxScalar = maxScalar;
//        leaf.minScalar = minScalar;
//        if (leaf.maxScalar < isovalue || leaf.minScalar > isovalue) {
//            leaf.type = Node_None;
//        }
//    }
//}
//
//template<class T, class B>
//__global__ void generateVertexIndices(Octree<T,B> *leafNodes, int size) {
//    unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//    unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
//    unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//    for (int i = offset; i < size; i += stride) {
//        auto root = leafNodes[i];
//        for (int e = 0; e < 12; e++) {
//            if (!root->representative[e]) {
//                continue;
//            }
//            auto vertex = root->representative[e];
//            vertex = findRepresentative(vertex);
//
//            vtkIdType id;
//            double p[3];
//            for (int j = 0; j < 3; j++) {
//                p[j] = vertex->position[j];
//            }
////            if (locator->InsertUniquePoint(p, id)) {
////                newScalars->InsertTuple(id, &root->isoValue);
////            }
//            vertex->index = id;
//        }
//    }
//}

void writeToFile(const std::string &i_file, vtkSmartPointer<vtkPolyDataAlgorithm> surface) {
    vtkNew<vtkOBJWriter> writer;
    writer->SetFileName(i_file.c_str());
    writer->SetInputConnection(surface->GetOutputPort());
    writer->Write();
}


int main() {
    ambiguousFacesInit();
    ambiguousCasesComplementInit();
    std::vector<std::pair<std::string, double>> files = {
            {"data/boston_teapot.nhdr", 20.5},
            {"data/skull.nhdr", 30},
            {"data/foot.nhdr", 110.3},
            {"data/vis_male.nhdr", 110.3},
            {"data/carp.nhdr",1150.5}
    };

    double adaptiveThreshold = 0;
    bool forceRewrite = false;
    for (auto &file : files) {
        auto i_file = file.first;
        double isoValue = file.second;
        std::cout << "Processing" << i_file << std::endl;
        vtkNew<vtkNrrdReader> nrrdReader;
        nrrdReader->SetFileName(i_file.c_str());
        nrrdReader->Update();
        vtkNew<vtkImageData> volume;
        volume->DeepCopy(nrrdReader->GetOutput());
        auto dims = volume->GetDimensions();
        vtkNew<myGPUMDMC> surface;
        surface->SetAdaptiveThreshold(adaptiveThreshold);
        surface->SetInputData(volume);
//    surface->ComputeNormalsOn();
        surface->ComputeNormalsOff();
        surface->SetValue(0, isoValue);


        std::string outputPath = "output";
        std::string consolePath = "console";

        if (!std::filesystem::exists(outputPath)) {
            std::filesystem::create_directories(outputPath);
        }
        if (!std::filesystem::exists(consolePath)) {
            std::filesystem::create_directories(consolePath);
        }
        std::string surfaceType = typeid(*surface).name();
        std::string fileName = std::filesystem::path(i_file).stem();
        std::string outputName = surfaceType + "_" + fileName  + "_"
                                 + std::to_string((int)isoValue) + "_" + std::to_string((int)adaptiveThreshold);
        auto consoleFilename = consolePath + "/" + outputName + ".txt";
        if (!forceRewrite && std::filesystem::exists(consoleFilename)) {
            std::cout << "file already exists, skip:" << consoleFilename << std::endl;
            continue;
        }

        std::ofstream consoleFile(consoleFilename);
        if (!consoleFile || !consoleFile.is_open()) {
            std::cout << "open console file fail" << std::endl;
            continue;
        }
        std::streambuf* originalCoutBuffer = std::cout.rdbuf();
        std::cout.rdbuf(consoleFile.rdbuf()); // 将输出重定向到consoleFile

        {
            surface->Update();
        }

        std::cout.rdbuf(originalCoutBuffer); // 将输出重定向回来
        consoleFile.close();
//        const std::string i_file = "test.obj";
        {
            std::string objName = outputPath + "/" + outputName + ".obj";
            writeToFile(objName, surface);
            std::cout << "Writing " << objName << std::endl;
        }


    }


    cudaFree(ambiguousCasesComplement);
    cudaFree(ambiguousFaces);



//    glm::u32vec3 size =  {dims[0] - 1, dims[1] - 1, dims[2] - 1};
//    int height = 0;
//    glm::u32vec3 conceptualSize = findLargerClosestPower2Vector(size, height);
//    std::cout << "max height:" << height << std::endl;
//    auto scalars = nrrdReader->GetOutput()->GetPointData()->GetScalars();
//    using ComponentRef =  typename vtk::detail::SelectValueRange<decltype(scalars), 1>::type;
//    using ArrayType  = decltype(scalars->GetArrayType());
//    using OctreeType = Octree<ArrayType, ComponentRef>;
//    auto scalarArray = vtk::DataArrayValueRange<1>(scalars);
//    auto numberOfTuples = scalars->GetNumberOfTuples();
//    VoxelsData<ArrayType> voxelsData = {.scalars = new ArrayType[numberOfTuples],
//            .cubeDims = size, .dims = {dims[0], dims[1], dims[2]}, .conceptualDims = conceptualSize };
//    for (int i = 0; i < numberOfTuples; i++) {
//        voxelsData.scalars[i] = scalarArray[i];
////        if (scalarArray[i])
////        std::cout << scalarArray[i] << std::endl;
//    }
//
//    VoxelsData<ArrayType> *deviceData;
//    cudaMallocManaged(&deviceData, sizeof(VoxelsData<ArrayType>));
//    cudaMemcpy(deviceData, &voxelsData, sizeof(VoxelsData<ArrayType>), cudaMemcpyHostToDevice);
//    ArrayType *deviceScalars;
//    cudaMallocManaged(&deviceScalars, sizeof(ArrayType) * numberOfTuples);
//    cudaMemcpy(deviceScalars, voxelsData.scalars, sizeof(ArrayType) * numberOfTuples, cudaMemcpyHostToDevice);
//    // set deviceScalars to deviceData.scalars
//    cudaMemcpy(&(deviceData->scalars), &deviceScalars, sizeof(ArrayType *), cudaMemcpyHostToDevice);
//    for (int i = 0; i < numberOfTuples; i++) {
//        if (deviceData->scalars[i] != voxelsData.scalars[i]) {
//            std::cout << "error" << std::endl;
//        }
//    }
////    deviceData->scalars = deviceScalars;
//
//    std::cout << voxelsData.scalars[0] << std::endl;
//    std::cout << voxelsData.dims[0] << " " << voxelsData.dims[1] << " " << voxelsData.dims[2] << std::endl;
//    std::cout << voxelsData.conceptualDims[0] << " " << voxelsData.conceptualDims[1] << " " << voxelsData.conceptualDims[2] << std::endl;
//    std::cout << voxelsData.cubeDims[0] << " " << voxelsData.cubeDims[1] << " " << voxelsData.cubeDims[2] << std::endl;
//
//    dim3 grid=(32,32);
//    dim3 block=(32,32);
////    dim3 block={16, 8, 8};
////    dim3 grid={conceptualSize.x / block.x, conceptualSize.y / block.y, conceptualSize.z / block.z};
//    // better grid and block for v100
//    auto start = std::chrono::high_resolution_clock::now();;
//    // transfer voxelsData to device
////    for (int i = 0; i < numberOfTuples; i++) {
////        if (deviceData->scalars[i] != 0) {
////            std::cout << deviceData->scalars[i] << std::endl;
////        }
////    }
//
//    OctreeType *leafNodes = nullptr;
//    {
//        Region region = {.origin = {0, 0, 0}, .size = size,
//                .voxelsCnt = size.x * size.y * size.z, .conceptualSize = conceptualSize};
//        int leafNum = size.x * size.y * size.z;
//        long long nByte = 1ll * sizeof(OctreeType) * leafNum;
//        cudaMallocManaged((void**)&leafNodes, nByte);
//        cudaMemset(leafNodes, 0, nByte);
//        generateLeafNodes<<<grid, block>>>(deviceData, leafNodes, leafNum, height, value);
//        auto error = cudaDeviceSynchronize();
//        // print the error
//        if (error != cudaSuccess) {
//            std::cout << cudaGetErrorString(error) << std::endl;
//        }
//        int cnt = 0;
//        for (int i = 0; i < leafNum; i++) {
//            auto &leaf = leafNodes[i];
//            unsigned x = i % voxelsData.cubeDims.x;
//            unsigned y = (i / voxelsData.cubeDims.x) % voxelsData.cubeDims.y;
//            unsigned z = i / (voxelsData.cubeDims.x * voxelsData.cubeDims.y);
////            std::cout << x << " " << y << " " << z << std::endl;
////            std::cout << leaf.region.origin.x << " " << leaf.region.origin.y << " " << leaf.region.origin.z << std::endl;
////            std::cout << position2Index(leaf.region.origin, voxelsData.dims) << std::endl;
////            std::cout << leaf.maxScalar << " " << leaf.minScalar << std::endl;
////            std::cout << leaf.normal[0].x << " " << leaf.normal[0].y << " " << leaf.normal[0].z << std::endl;
//            if (leaf.type == Node_None) {
//                continue;
//            }
//            OctreeType::calculateMDCRepresentative(&leaf, &leaf, value, 1);
//            cnt++;
//        }
//        std::cout << "leaf node num:" << cnt << std::endl;
//    }
//
//
//    OctreeType **everyHeightNodes = new OctreeType*[height+1];
//    everyHeightNodes[0] = leafNodes;
//    unsigned regionSize = 1;
//    glm::u32vec3 childrenSize = size;
//    for (int h = 1; h <= height; h++) {
//        OctreeType *nodes = nullptr;
//        regionSize *= 2;
//        // roundUp
//        glm::u32vec3 curSize = (childrenSize + 1u) / 2u;
//        int nodeNums = curSize.x * curSize.y * curSize.z;
//        long long nByte = 1ll * sizeof(OctreeType) * nodeNums;
//        cudaMallocManaged((void**)&nodes, nByte);
//        cudaMemset(nodes, 0, nByte);
//        std::cout << nodeNums << std::endl;
//        generateInternalNodes<<<grid, block>>>(deviceData, nodes, everyHeightNodes[h-1], nodeNums, height - h, h, value, childrenSize, curSize, regionSize);
//        cudaDeviceSynchronize();
//        everyHeightNodes[h] = nodes;
//        for (int i = 0; i < nodeNums; i++) {
//            auto &node = nodes[i];
//            if (node.type == Node_None) {
//                continue;
//            }
//            1 == 1;
////            std::cout << node.region.origin.x << " " << node.region.origin.y << " " << node.region.origin.z << std::endl;
//        }
//        childrenSize = curSize;
//    }
//
//    OctreeType *root = everyHeightNodes[height];
//
//
//auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> duration = end - start;
//    double seconds = duration.count();
//    std::cout << "函数执行时间: " << seconds << " 秒" << std::endl;
//
//
//    cudaFree(deviceData);
//    delete voxelsData.scalars;
//    // free the everyHeightNodes
//    for (int i = 0; i < height; i++) {
//        cudaFree(everyHeightNodes[i]);
//    }
//    delete []everyHeightNodes;

    return 0;
}
