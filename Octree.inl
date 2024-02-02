//
// Created by hougr t on 2023/11/17.
//
#pragma once
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "common.cuh"
#include "table.cuh"
//#include "nlopt.hpp"
#include "UnionFind.h"


namespace {
    const float QEF_ERROR = 1e-6f;
    const int QEF_SWEEPS = 4;
    const int edges2Vertices[12][2] = {{0, 1},
                                 {1, 3},
                                 {2, 3},
                                 {0, 2},
                                 {4, 5},
                                 {5, 7},
                                 {6, 7},
                                 {4, 6},
                                 {0, 4},
                                 {1, 5},
                                 {3, 7},
                                 {2, 6}};
    __device__ const int device_edges2Vertices[12][2] = {{0, 1},
                                       {1, 3},
                                       {2, 3},
                                       {0, 2},
                                       {4, 5},
                                       {5, 7},
                                       {6, 7},
                                       {4, 6},
                                       {0, 4},
                                       {1, 5},
                                       {3, 7},
                                       {2, 6}};
    // 共面的两个children(首次)
    const int cellProcFaceMask[12][3] = {{0, 1, 0},
                                         {4, 5, 0},
                                         {2, 3, 0},
                                         {6, 7, 0},
                                         {0, 2, 1},
                                         {1, 3, 1},
                                         {4, 6, 1},
                                         {5, 7, 1},
                                         {0, 4, 2},
                                         {2, 6, 2},
                                         {1, 5, 2},
                                         {3, 7, 2}};
    // 共线的四个children(首次)
    const int cellProcEdgeMask[6][5] = {{0, 4, 2, 6, 0},
                                        {1, 5, 3, 7, 0},
                                        {0, 1, 4, 5, 1},
                                        {2, 3, 6, 7, 1},
                                        {0, 2, 1, 3, 2},
                                        {4, 6, 5, 7, 2}};

    // 共面的两个children+方向(两个共面之中的)
    const int faceProcFaceMask[3][4][3] = {
            {{1, 0, 0}, {5, 4, 0}, {3, 2, 0}, {7, 6, 0}},
            {{2, 0, 1}, {3, 1, 1}, {6, 4, 1}, {7, 5, 1}},
            {{4, 0, 2}, {6, 2, 2}, {5, 1, 2}, {7, 3, 2}}
    };

    // order+共线的四个children+方向(两个共面之中的)
    const int faceProcEdgeMask[3][4][6] = {
            {{1, 1, 0, 5, 4, 1}, {1, 3, 2, 7, 6, 1}, {0, 1, 3, 0, 2, 2}, {0, 5, 7, 4, 6, 2}},
            {{0, 2, 6, 0, 4, 0}, {0, 3, 7, 1, 5, 0}, {1, 2, 0, 3, 1, 2}, {1, 6, 4, 7, 5, 2}},
            {{1, 4, 0, 6, 2, 0}, {1, 5, 1, 7, 3, 0}, {0, 4, 5, 0, 1, 1}, {0, 6, 7, 2, 3, 1}}
    };


    // 共线的四个children+方向(四个共线的cube)
     const int edgeProcEdgeMask[3][2][5] = {
            {{6, 2, 4, 0, 0}, {7, 3, 5, 1, 0}},
            {{5, 4, 1, 0, 1}, {7, 6, 3, 2, 1}},
            {{3, 1, 2, 0, 2}, {7, 5, 6, 4, 2}},
    };

    // 共线在四个cube之中的编号
    const int processEdgeMask[3][4] = {{6,  2, 4,  0},
                                       {5,  7, 1,  3},
                                       {10, 9, 11, 8}};

    __device__ const int device_processEdgeMask[3][4] = {{6,  2, 4,  0},
                                       {5,  7, 1,  3},
                                       {10, 9, 11, 8}};
    const int directionCube[][6] = {
            // top,bottom,left,right,front,back
            {4,  14, 11, 1,  12, 2},
            {5,  15, 0,  10, 13, 3},
            {6,  16, 13, 3,  0,  10},
            {7,  17, 2,  12, 1,  11},
            {10, 0,  15, 5,  16, 6},
            {11, 1,  4,  14, 17, 7},
            {12, 2,  17, 7,  4,  14},
            {13, 3,  6,  16, 5,  15},
    };


    template<class T>
    __host__ __device__ void vtkMarchingCubesComputePointGradient(
            glm::u32vec3 pos, const T &s, glm::u32vec3 dims, int sliceSize, double n[3]) {
        auto i = pos[0], j = pos[1], k = pos[2];
        double sp, sm;
        // x-direction
        if (i == 0) {
            sp = s[i + 1 + j * dims[0] + k * sliceSize];
            sm = s[i + j * dims[0] + k * sliceSize];
            n[0] = sm - sp;
        } else if (i == (dims[0] - 1)) {
            sp = s[i + j * dims[0] + k * sliceSize];
            sm = s[i - 1 + j * dims[0] + k * sliceSize];
            n[0] = sm - sp;
        } else {
            sp = s[i + 1 + j * dims[0] + k * sliceSize];
            sm = s[i - 1 + j * dims[0] + k * sliceSize];
            n[0] = 0.5 * (sm - sp);
        }

        // y-direction
        if (j == 0) {
            sp = s[i + (j + 1) * dims[0] + k * sliceSize];
            sm = s[i + j * dims[0] + k * sliceSize];
            n[1] = sm - sp;
        } else if (j == (dims[1] - 1)) {
            sp = s[i + j * dims[0] + k * sliceSize];
            sm = s[i + (j - 1) * dims[0] + k * sliceSize];
            n[1] = sm - sp;
        } else {
            sp = s[i + (j + 1) * dims[0] + k * sliceSize];
            sm = s[i + (j - 1) * dims[0] + k * sliceSize];
            n[1] = 0.5 * (sm - sp);
        }

        // z-direction
        if (k == 0) {
            sp = s[i + j * dims[0] + (k + 1) * sliceSize];
            sm = s[i + j * dims[0] + k * sliceSize];
            n[2] = sm - sp;
        } else if (k == (dims[2] - 1)) {
            sp = s[i + j * dims[0] + k * sliceSize];
            sm = s[i + j * dims[0] + (k - 1) * sliceSize];
            n[2] = sm - sp;
        } else {
            sp = s[i + j * dims[0] + (k + 1) * sliceSize];
            sm = s[i + j * dims[0] + (k - 1) * sliceSize];
            n[2] = 0.5 * (sm - sp);
        }
    };


    auto isInBigCube(int x, int y, int z, glm::u32vec3 dims) {
        return x >= 0 && x < dims[0] && y >= 0 && y < dims[1] && z >= 0 && z < dims[2];
    };


//    template <class T>
//    auto checkSubdivide(int x, int y, int z, int size, int cubeDims[3], double isovalue, T scalars) {
//        // edge ambiguity
//        int halfSize = size;
//        for (int e = 0; e < 12; e++) {
//            int v0 = edges2Vertices[e][0];
//            int v1 = edges2Vertices[e][1];
//            auto pos0 = localVerticesPos[v0];
//            auto pos1 = localVerticesPos[v1];
//            int axisChange[3] = {pos0[0] ^ pos1[0], pos0[1] ^ pos1[1], pos0[2] ^ pos1[2]};
//            int ambiguous = 0;
//            for (int i = 0; i < halfSize - 1; i++) {
//                int cv0[3] = {x + pos0[0] * halfSize + i * axisChange[0], y + pos0[1] * halfSize + i * axisChange[1],
//                              z + pos0[2] * halfSize + i * axisChange[2]};
//                int cv1[3] = {x + pos1[0] * halfSize + (i + 1) * axisChange[0], y + pos1[1] * halfSize + (i + 1) * axisChange[1],
//                              z + pos1[2] * halfSize + (i + 1) * axisChange[2]};
//                if (!isInBigCube(cv0[0], cv0[1], cv0[2], cubeDims) || !isInBigCube(cv1[0], cv1[1], cv1[2], cubeDims)) {
//                    break;
//                }
//                auto scalar0 = scalars[position2Index(cv0[0], cv0[1], cv0[2], cubeDims)];
//                auto scalar1 = scalars[position2Index(cv1[0], cv1[1], cv1[2], cubeDims)];
//                bool sign0 = scalar0 >= isovalue;
//                bool sign1 = scalar1 >= isovalue;
//                if (sign0 ^ sign1) {
//                    ambiguous++;
//                }
//                if (ambiguous >= 2) {
//                    return true;
//                }
//            }
//        }
//        // complex surface
//        double normals[8][3];
//        for (int i = 0; i < 8; i++) {
//            int order[3] = {localVerticesPos[i][0] * halfSize + x, localVerticesPos[i][1] * halfSize + y, localVerticesPos[i][2] * halfSize + z};
//            vtkMarchingCubesComputePointGradient(order[0], order[1], order[2], scalars, cubeDims, cubeDims[0] * cubeDims[1], normals[i]);
//        }
//
//
//        return false;
//    };


//    auto addEdges = [&] (const int rootIndex[3]) {
//        for (int i = 0; i < 3; i++) {
//            for (int j = i+1; j < 3; j++) {
//                auto r1 = representatives[rootIndex[i]];
//                auto r2 = representatives[rootIndex[j]];
//                auto left = reinterpret_cast<uintptr_t >(r1);
//                auto right = reinterpret_cast<uintptr_t >(r2);
//                std::pair<uintptr_t , uintptr_t> pair;
//                if (left < right) {
//                    pair = std::make_pair(left, right);
//                } else {
//                    pair = std::make_pair(right, left);
//                }
//
//                auto other = representatives[rootIndex[3- i -j]];
//                edgeMap[pair].insert(other);
//                if (!other->canMerge) {
//                    other->canMerge;
//                }
//                    if (edgeMap[pair].size() > 2) {
////                        std::cout << "non manifold edge found!"  << std::endl;
//                        nonManifoldEdges.insert(pair);
//                        if (ambiguousCasesComplement.find(r1->sign) == ambiguousCasesComplement.end() && ambiguousCasesComplement.find(r2->sign) == ambiguousCasesComplement.end()) {

//                        nonManifoldEdges.insert(pair);
//                        if (ambiguousCasesComplement.find(r1->sign) == ambiguousCasesComplement.end() && ambiguousCasesComplement.find(r2->sign) == ambiguousCasesComplement.end()) {
//                            std::cout << "non manifold edge found!"  << std::endl;
//
//                        }
//                        std::cout << "non manifold edge:" << nonManifoldEdges.size() << std::endl;
//                    }
//            }
//        }
//    };

    __host__ __device__ bool inRegion(glm::vec3 pos, const Region &region) {
        return pos.x >= region.origin.x && pos.x < region.origin.x + region.size.x &&
               pos.y >= region.origin.y && pos.y < region.origin.y + region.size.y &&
               pos.z >= region.origin.z && pos.z < region.origin.z + region.size.z;
    }

    __host__ __device__ Direction convertToDirection(const glm::vec3 &dir) {
        Direction direction;
        if (dir[0] == -1) {
            direction = Direction::LEFT;
        } else if (dir[0] == 1) {
            direction = Direction::RIGHT;
        } else if (dir[1] == -1) {
            direction = Direction::FRONT;
        } else if (dir[1] == 1) {
            direction = Direction::BACK;
        } else if (dir[2] == -1) {
            direction = Direction::BOTTOM;
        } else if (dir[2] == 1) {
            direction = Direction::TOP;
        }
        return direction;
    }

    __host__ __device__ int convertToFace(Direction dir) {
        int face;
        switch (dir) {
            case Direction::BOTTOM:
                face = 0;
                break;
            case Direction::TOP:
                face = 1;
                break;
            case Direction::FRONT:
                face = 2;
                break;
            case Direction::BACK:
                face = 3;
                break;
            case Direction::LEFT:
                face = 4;
                break;
            case Direction::RIGHT:
                face = 5;
                break;
            default:
                face = -1;
                break;
        }
        return face;
    }

    __host__ __device__ Direction convertToDirection(int face) {
        Direction direction;
        switch (face) {
            case 0:
                direction = Direction::BOTTOM;
                break;
            case 1:
                direction = Direction::TOP;
                break;
            case 2:
                direction = Direction::FRONT;
                break;
            case 3:
                direction = Direction::BACK;
                break;
            case 4:
                direction = Direction::LEFT;
                break;
            case 5:
                direction = Direction::RIGHT;
                break;
            default:
                direction = Direction::DIRECTION_NUM;
                break;
        }
        return direction;
    }
    __host__ __device__ Direction reverseDirection(Direction dir) {
        return static_cast<Direction>(dir ^ 0x1);
    }

    __host__ __device__ OctreeRepresentative * findRepresentative(OctreeRepresentative * vertex) {
        if (!vertex) {
            return nullptr;
        }
        auto correctVertex = vertex;
        while (vertex) {
            // used for manifold dmc, normal dmc has no effect due to the nullptr parent
//            if (!vertex->canMerge) {
//                break;
//            }
            if (vertex->collapsible) {
                correctVertex = vertex;
            }
            vertex = vertex->parent;
        }
        return correctVertex;
    }

    OctreeRepresentative* findCommonAncestor(OctreeRepresentative* v1, OctreeRepresentative* v2) {
        if (!v1 || !v2) {
            return nullptr;
        }
        if (v1 == v2) {
            return v1;
        }
        auto ancestor1 = v1;
        auto ancestor2 = v2;
        while (ancestor1 && ancestor2 && ancestor1 != ancestor2) {
            if (ancestor1->height < ancestor2->height) {
                ancestor1 = ancestor1->parent;
            } else if (ancestor1->height > ancestor2->height) {
                ancestor2 = ancestor2->parent;
            } else {
                ancestor1 = ancestor1->parent;
                ancestor2 = ancestor2->parent;
            }
        }
        if (!ancestor1 || !ancestor2) {
            return nullptr;
        }
        return ancestor1;
    }

    __host__ __device__ bool isAmbiguousCase(uint8_t sign) {
        return ambiguousCasesComplement[sign];
    }

    double objective(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
        const std::vector<double> &b = *reinterpret_cast<std::vector<double>*>(my_func_data);
        // 计算梯度，如果grad不为空
        if (!grad.empty()) {
            grad[0] = 2 * (x[0] - b[0]);
            grad[1] = 2 * (x[1] - b[1]);
            grad[2] = 2 * (x[2] - b[2]);
        }

        // 返回距离
        return std::pow(x[0] - b[0], 2) + std::pow(x[1] - b[1], 2) + std::pow(x[2] - b[2], 2);
    }

    template<class T, class B>
    double constraint(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        // 假设f(x)已经定义，并且k是一个给定的常数
//        double k = *reinterpret_cast<double*>(data);
        Octree<T,B> *root = reinterpret_cast<Octree<T,B>*>(data);

        double k = root->isoValue;

        // 计算f(x)
        auto scalars = root->scalar;
        double f[8];
        for (int i = 0; i < 8; i++) {
            f[i] = scalars[i];
        }
        double fx = trilinearInterpolation(scalars, {x[0], x[1], x[2]});
        if (!grad.empty()) {
//            auto g = trilinearInterpolation(root->normal, {x[0], x[1], x[2]});
            auto u = x[0];
            auto v = x[1];
            auto w = x[2];
            double gradU = (1-w)*((f[1]-f[0])*(1-v) + (f[3] - f[2]) * v) +
                           w*((f[5]-f[4])*(1-v) + (f[7] - f[6]) * v);
            double gradV = (1-w)*((f[2]-f[0])*(1-u) + (f[3] - f[1]) * u) +
                           w*((f[6]-f[4])*(1-u) + (f[7] - f[5]) * u);
            double gradW = (1-v)*((f[4]-f[0])*(1-u) + (f[5] - f[1]) * u) +
                           v*((f[6]-f[2])*(1-u) + (f[7] - f[3]) * u);
            gradU *= 2*(fx - k);
            gradV *= 2*(fx - k);
            gradW *= 2*(fx - k);
//            grad[0] = g[0];
//            grad[1] = g[1];
//            grad[2] = g[2];
//            if (fx >= k) {
//                grad[0] = gradU;
//                grad[1] = gradV;
//                grad[2] = gradW;
//            } else {
//                grad[0] = -gradU;
//                grad[1] = -gradV;
//                grad[2] = -gradW;
//            }
            grad[0] = gradU;
            grad[1] = gradV;
            grad[2] = gradW;
        }

        // 这里不计算梯度
//        return fabs(fx - k);
        return (fx - k) * (fx - k);
    };
}

namespace std {
    template<>
    struct hash<glm::u32vec3> {
        size_t operator()(const glm::u32vec3& vec) const {
            // 实现你的哈希函数
            // 你可以将三个分量组合起来生成一个哈希值
            // 例如，使用位运算和一些质数来减少哈希碰撞的可能性
            return std::hash<unsigned int>()(vec.x) ^
                   (std::hash<unsigned int>()(vec.y) << 1) ^
                   (std::hash<unsigned int>()(vec.z) << 2);
        }
    };

}


template<class T, class B>
bool Octree<T,B>::invertSign(Octree *root, Octree *templateRoot, double isovalue) {
    auto sign = root->sign;
    if (!isAmbiguousCase(sign)) {
        return false;
    }

    if (!templateRoot) {
        return false;
    }

    auto dir = ambiguousFaces[~sign];
    Direction direction = convertToDirection(dir);
    // new tree has not been built completely, so use template tree instead
    auto neighbor = findNeighbor(templateRoot, direction);
    if (neighbor) {
        auto neighborSign = 0;
        for (int i = 0; i < 8; i++) {
            neighborSign |= (neighbor->scalar[i] >= isovalue) ? 1 << i : 0;
        }
        if (isAmbiguousCase(neighborSign)) {
            return true;
        }
    }
    return false;
}

template<class T, class B>
bool Octree<T, B>::calculateMDCRepresentative(Octree *root, Octree *templateRoot, OctreeRepresentative *allVertex, int *globalIndex, double isovalue, int useOptimization) {
    if (!root) {
        return false;
    }
    if (!isLeaf(root)) {
        return false;
    }


    auto sign = root->sign;
    bool ambiguous = isAmbiguousCase(sign);
    if (ambiguous && !templateRoot) {
        return false;
    }

//    bool nonManifold = false;
    int clusterDisable = 0;

    if (ambiguous && invertSign(root, templateRoot, isovalue)) {
        sign = ~sign;
    }

    if (!ambiguous) {
        for (int f = 0; f < 6; f++) {
            auto vs = v_face[f];
            int flag = 0;
            for (int i = 0; i < 4; i++) {
                if (sign & (1 << vs[i])) {
                    flag |= (1 << i);
                }
            }
            // ambiguous face
            if (flag != 0b1001 && flag != 0b0110) {
                continue;
            }
            if (!templateRoot) {
                return false;
            }
            auto dir = convertToDirection(f);
            auto neighbor = findNeighbor(templateRoot, dir);
            if (!neighbor) {
                continue;
            }
            auto neighborSign = 0;
            for (int i = 0; i < 8; i++) {
                neighborSign |= (neighbor->scalar[i] >= isovalue) ? 1 << i : 0;
            }
            if (isAmbiguousCase(neighborSign)) {
                for (int i = 0; i < 4; i++) {
                    auto edges = e_face[f];
                    clusterDisable |= (1 << edges[i]);
                }
            }
        }
    }

    if (useOptimization == 1) {
        calculateAccurateRepresentative(root, allVertex, globalIndex, isovalue, sign, clusterDisable);
    }
    return true;

}

template <class T, class B>
void Octree<T, B>::calculateAccurateRepresentative(Octree *root, OctreeRepresentative *allVertex, int *globalIndex, double isovalue, uint8_t sign, int clusterDisable) {
    auto contoursTable = &r_pattern[(int)sign * 17];
    auto numContours = contoursTable[0];
    auto contourBegin = numContours + 1;
    const uint w_proj = 076543210;
    const uint v_proj = 076325410;
    const uint u_proj = 075316420;
    enum ProjectionDirection { W_PROJECTION = 1, V_PROJECTION = 2, U_PROJECTION = 3};
    // collect data
    glm::vec3 bboxMin[4] ={ { 1,1, 1}, { 1,1, 1}, { 1,1, 1}, { 1,1, 1}};
    glm::vec3 bboxMax[4] = { { 0,0, 0}, { 0,0, 0}, { 0,0, 0}, { 0,0, 0}};
    auto origin = root->region.origin;
    auto lProjection = [&] (const glm::vec3& ni)
    {
        const double dx = std::fabs(ni.x);
        const double dy = std::fabs(ni.y);
        const double dz = std::fabs(ni.z);
        if (dz >= dx && dz >= dy) return ProjectionDirection::W_PROJECTION;
        if (dy >= dx && dy >= dz) return ProjectionDirection::V_PROJECTION;
        if (dx >= dy && dx >= dz) return ProjectionDirection::U_PROJECTION;
        return ProjectionDirection::W_PROJECTION;
    };

    auto minProjection = [&] (const double u, const double v, const double w)
    {
        if (w <= u && w <= v) return ProjectionDirection::W_PROJECTION;
        if (v <= u && v <= w) return ProjectionDirection::V_PROJECTION;
        if (u <= v && u <= w) return ProjectionDirection::U_PROJECTION;
        return ProjectionDirection::W_PROJECTION;
    };

    auto isInNeighborBBox = [&] (const glm::vec3 pos, const int contourIndex, bool onlyTriangle = true)
    {
        for (int i = 0; i < numContours; i++) {
            const int contourVertexNum = contoursTable[i + 1];
            if ((!onlyTriangle || contourVertexNum == 3) && contourIndex != i) {
                if (pos.x < bboxMin[i].x || bboxMax[i].x < pos.x) {
                    continue;
                }
                if (pos.y < bboxMin[i].y || bboxMax[i].y < pos.y) {
                    continue;
                }
                if (pos.z < bboxMin[i].z || bboxMax[i].z < pos.z) {
                    continue;
                }
                return true;
            }
        }
        return false;
    };

    auto projection = [&] (ProjectionDirection projectionDirection,
                           const int contourIndex, const int nrSamples, const glm::vec3 &massPoint) {
        const glm::vec3 &boxMin = bboxMin[contourIndex];
        const glm::vec3 &boxMax = bboxMax[contourIndex];
        int contourVertexNum = contoursTable[contourIndex + 1];
        glm::vec2 uvMin, uvMax;
        glm::vec2 wRange;
        // vertex index of the projection direction
        uint vIndex = w_proj;
        glm::vec3 p = massPoint;
        switch (projectionDirection) {
            case W_PROJECTION:
                uvMin = {boxMin.x, boxMin.y};
                uvMax = {boxMax.x, boxMax.y};
                wRange = {boxMin.z, boxMax.z};
                p = massPoint;
                vIndex = w_proj;
                break;
            case V_PROJECTION:
                uvMin = {boxMin.x, boxMin.z};
                uvMax = {boxMax.x, boxMax.z};
                wRange = {boxMin.y, boxMax.y};
                p = {massPoint.x, massPoint.z, massPoint.y};
                vIndex = v_proj;
                break;
            case U_PROJECTION:
                uvMin = {boxMin.y, boxMin.z};
                uvMax = {boxMax.y, boxMax.z};
                wRange = {boxMin.x, boxMax.x};
                p = {massPoint.y, massPoint.z, massPoint.x};
                vIndex = u_proj;
                break;
        }
        const double du = (uvMax.x - uvMin.x) / (nrSamples - 1);
        const double dv = (uvMax.y - uvMin.y) / (nrSamples - 1);
        float minDistance = 100;
        const double eps{ 1e-5 };
        auto f = root->scalar;
        auto r = vIndex;
        if (std::fabs(wRange.y - wRange.x) < eps) {
            wRange = {wRange.x - eps, wRange.y + eps};
        }

        glm::vec3 pt = {-1, -1, -1};
        for (int i = 1; i < (nrSamples - 1); i++) {
            float u = uvMin.x + i * du;
            for (int j = 1; j < (nrSamples - 1); j++) {
                float v = uvMin.y + j * dv;
                const float g1 = (1 - v) * ((1 - u) * f[0] + u * f[(r >> 3) & 0x7]) + v * ((1 - u) * f[(r >> 6) & 0x7] + u * f[(r >> 9) & 0x7]);
                const float g2 = (1 - v) * ((1 - u) * f[(r >> 12) & 0x7] + u * f[(r >> 15) & 0x7]) + v * ((1 - u) * f[(r >> 18) & 0x7] + u * f[(r >> 21) & 0x7]);
                if (g1 == g2) continue;
                float w = (isovalue - g1) / (g2 - g1);
                if (wRange.x > w || w > wRange.y) {
                    continue;
                }
                glm::vec3 pos;
                switch (projectionDirection) {
                    case W_PROJECTION:
                        pos = {u, v, w};
                        break;
                    case V_PROJECTION:
                        pos = {u, w, v};
                        break;
                    case U_PROJECTION:
                        pos = {w, u, v};
                        break;
                }
                // 三角形轮廓直接判断，否则判断是否在别的轮廓里
                if (numContours == 1 || contourVertexNum == 3 || !isInNeighborBBox(pos, contourIndex))
                {
                    glm::vec3 curPos = {u, v, w};
                    auto d = glm::distance(curPos, p);
                    // TODO 为什么下面这行不行
//                   float d = (p[0] - u) * (p[0] - u) + (p[1] - v) * (p[1] - v) + (p[2] - w) * (p[2] - w);
                    if (minDistance  > d) {
                        minDistance = d;
                        pt = pos;
//                      auto error = trilinearInterpolation(root->scalar, pt);
//                      error = error;
                    }
                }
            }
        }
        return pt;
    };


    auto calRepresentative = [&] (int contourIndex, const glm::vec3 &gradient, const glm::vec3 &massPoint) {
        const int nrSamples{ 9 };
        int contourVertexNum = contoursTable[contourIndex + 1];
        ProjectionDirection prj{ ProjectionDirection::W_PROJECTION };
        if (contourVertexNum >= 6) {
            prj = lProjection(gradient);
        } else {
            prj = minProjection(bboxMax[contourIndex].x - bboxMin[contourIndex].x,
                                bboxMax[contourIndex].y - bboxMin[contourIndex].y,
                                bboxMax[contourIndex].z - bboxMin[contourIndex].z);
        }

//        glm::vec3 pt = massPoint;
        glm::vec3 pt = projection(prj, contourIndex, nrSamples, massPoint);
        return pt;
    };

//    auto optimize = [&](int contourIndex, const glm::vec3 &gradient, const glm::vec3 &massPoint) {
//        nlopt::opt opt(nlopt::LD_AUGLAG, 3);
//        std::vector<double> mp = {massPoint.x, massPoint.y, massPoint.z};
//        const double tol = 1e-5;
//        const double xtol = 1e-8;
//        const double ftol = 1e-6;
//        const double etol = 1e-6;
//        // 约束函数：f(x) - k 应该等于 0
////        opt.set_min_objective(objective, &mp);
////        opt.add_equality_constraint(constraint<T, B>, root, etol);
////        opt.add_equality_constraint(objective, &mp);
//        opt.set_min_objective(constraint<T, B>, root);
//        auto minBox = bboxMin[contourIndex];
//        auto maxBox = bboxMax[contourIndex];
//        if (maxBox.x - minBox.x < tol) {
//            maxBox.x += tol;
//            minBox.x -= tol;
//        }
//        if (maxBox.y - minBox.y < tol) {
//            maxBox.y += tol;
//            minBox.y -= tol;
//        }
//        if (maxBox.z - minBox.z < tol) {
//            maxBox.z += tol;
//            minBox.z -= tol;
//        }
//        maxBox = glm::min(maxBox, glm::vec3(1, 1, 1));
//        minBox = glm::max(minBox, glm::vec3(0, 0, 0));
//        opt.set_xtol_abs(xtol);
////        opt.set_ftol_abs(ftol);
////        opt.set_xtol_rel(1e-4);
////        opt.set_stopval(1e-6);
//        opt.set_maxeval(1000);
////        opt.set_maxtime(0.1);
//        opt.set_lower_bounds({minBox.x, minBox.y, minBox.z});
//        opt.set_upper_bounds({maxBox.x, maxBox.y, maxBox.z});
//        std::vector<double> x = {massPoint.x, massPoint.y, massPoint.z};
//        double minf;
//        nlopt::result result;
//        glm::vec3 res;
//        result = opt.optimize(x, minf);
//        res = {x[0], x[1], x[2]};
//
//        const int contourVertexNum = contoursTable[contourIndex + 1];
//        if (numContours == 1 || contourVertexNum == 3 || !isInNeighborBBox(res, contourIndex, false)) {
//            return res;
//        } else {
//            return massPoint;
//        }
////        return res;
//    };


    glm::vec3 localMassPoints[4];
    glm::vec3 localNormals[4];
    OctreeRepresentative* representatives[4];
    int index = atomicAdd(globalIndex, numContours);

    for (int i = 1; i <= numContours; i++) {
        int numsEdges = contoursTable[i];
        glm::vec3 contourP;
        glm::vec3 contourN;
        svd::QefSolver qefSolver;
        glm::vec3 averageNormal(0, 0, 0);
        int contourIndex = i - 1;
        bool canMerge = true;
        int edgeIntersection[12];
        for (int j = 0; j < 12; j++) {
            edgeIntersection[j] = 0;
        }
        for (int j = 0; j < numsEdges; j++) {
            auto e = contoursTable[contourBegin + j];
            int v0 = device_edges2Vertices[e][0];
            int v1 = device_edges2Vertices[e][1];
            auto s1 = root->scalar[v0];
            auto s2 = root->scalar[v1];
            bool sign1 = (root->sign >> v0) & 1;
            bool sign2 = (root->sign >> v1) & 1;
            auto pos1 = device_localVerticesPos[v0];
            auto pos2 = device_localVerticesPos[v1];
            auto n1 = root->normal[v0];
            auto n2 = root->normal[v1];
            glm::vec3 p1 = pos1;
            glm::vec3 p2 = pos2;
            float t = (isovalue - s1) / (s2 - s1);
            glm::vec3 p = p1 + t * (p2 - p1);
            glm::vec3 n = n1 + t * (n2 - n1);
            averageNormal += n;
            edgeIntersection[e] = 1;
            bboxMin[contourIndex] = glm::min(bboxMin[contourIndex], p);
            bboxMax[contourIndex] = glm::max(bboxMax[contourIndex], p);
            p += origin;
            qefSolver.add(p.x, p.y, p.z, n[0], n[1], n[2]);

            if (clusterDisable && ((clusterDisable >> e) & 0x1)) {
                canMerge = false;
            }
        }

        if (numsEdges != 0) {
            averageNormal /= numsEdges;
        }
        svd::Vec3 qefPosition;
        qefSolver.solve(qefPosition, QEF_ERROR, QEF_SWEEPS, QEF_ERROR);
        const auto &mp = qefSolver.getMassPoint();
        glm::vec3 massPoint = {mp.x, mp.y, mp.z};
        glm::vec3 localMassPoint = massPoint - glm::vec3(origin);
        localMassPoints[i-1] = localMassPoint;
        localNormals[i-1] = averageNormal;
        glm::vec3 qefPos = {qefPosition.x, qefPosition.y, qefPosition.z};
        if (!inRegion(qefPos, root->region)) {
            qefPos = massPoint;
        } else {
            qefPos = qefPos;
        }
        // TODO choose qefPos or pt
        OctreeRepresentative *representative = &allVertex[index + i - 1];
        *representative = OctreeRepresentative();
//        OctreeRepresentative *representative = allVertex + 4 * globalIndex + i - 1;
        representative->averageNormal = glm::normalize(averageNormal);
        representative->qef = qefSolver.getData();
        representative->position = qefPos;
        representative->cellIndex = root->index;
        representative->sign = sign;
        representative->canMerge = canMerge;
        representative->layerId = index + i - 1;
        for (int j = 0; j < 12; j++) {
            representative->edgeIntersection[j] = edgeIntersection[j];
        }
        representatives[i-1] = representative;
        for (int j = 0; j < numsEdges; j++) {
            auto e = contoursTable[contourBegin + j];
            root->representative[e] = representative;
        }
        contourBegin += numsEdges;
    }

    for (int i = 0; i < numContours; i++) {
        auto v = representatives[i];
        auto localMassPoint = localMassPoints[i];
        auto averageNormal = localNormals[i];
//        glm::vec3 pt = optimize(i, averageNormal, localMassPoint);
        glm::vec3 pt = calRepresentative(i, averageNormal, localMassPoint);
//        glm::vec3 pt = localMassPoint;
        const int contourVertexNum = contoursTable[i + 1];
        if (pt.x < 0 || pt.y < 0 || pt.z < 0) {
//            pt = optimize(i, averageNormal, localMassPoint);
            pt = localMassPoint;

//            pt = calRepresentative(i, averageNormal, localMassPoint);
//            if (pt.x < 0 || pt.y < 0 || pt.z < 0) {
//                pt = localMassPoint;
//            }
        }
//        if (numContours == 1 || contourVertexNum == 3 || !isInNeighborBBox(pt, i, false)) {
//        } else {
//            pt = localMassPoint;
//        }
        v->position = pt + glm::vec3(origin);
//        pt.x += origin.x;
//        pt.y += origin.y;
//        pt.z += origin.z;
//        v->position = pt;

    }
}


template<class T, class B>
Octree<T,B> * Octree<T, B>::findNeighbor(Octree *root, Direction dir) {
    if (!root || !root->parent) {
        return nullptr;
    }
    return root->neighbour[dir];
}


template <class T, class B>
void Octree<T, B>::generateVerticesIndices(Octree<T, B> *root,  vtkIncrementalPointLocator *locator, vtkDataArray* newScalars) {
    if (!root) {
        return;
    }
    if (isLeaf(root)) {
        for (int e = 0; e < 12; e++) {
            if (!root->representative[e]) {
                continue;
            }
            auto vertex = root->representative[e];
            vertex = findRepresentative(vertex);

            vtkIdType id;
            double p[3];
            for (int i = 0; i < 3; i++) {
                p[i] = vertex->position[i];
            }
            if (locator->InsertUniquePoint(p, id)) {
                newScalars->InsertTuple(id, &root->isoValue);
            }
            vertex->index = id;
        }
        return;
    }

    for (int i = 0; i < 8; i++) {
        generateVerticesIndices(root->children[i], locator, newScalars);
    }
}

template <class T, class B>
void Octree<T, B>::contourCellProc(Octree<T, B> *root, vtkCellArray* newPolys, int useOptimization) {
    if (!root) {
        return;
    }

    if (isLeaf(root)) {
        return;
    }

    if (root->collapsible) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        contourCellProc(root->children[i], newPolys, useOptimization);
    }

    for (int i = 0; i < 12; i++) {
        Octree<T, B> *nodes[2];
        const int c[2] = {cellProcFaceMask[i][0], cellProcFaceMask[i][1]};
        nodes[0] = root->children[c[0]];
        nodes[1] = root->children[c[1]];
        contourFaceProc(nodes, cellProcFaceMask[i][2], newPolys, useOptimization);
    }

    for (int i = 0; i < 6; i++) {
        Octree<T, B> *nodes[4];
        const int c[4] = {cellProcEdgeMask[i][0], cellProcEdgeMask[i][1], cellProcEdgeMask[i][2], cellProcEdgeMask[i][3]};
        for (int j = 0; j < 4; j++) {
            nodes[j] = root->children[c[j]];
        }
        contourEdgeProc(nodes, cellProcEdgeMask[i][4], newPolys, useOptimization);
    }
}

template<class T, class B>
void Octree<T, B>::contourFaceProc(Octree<T, B> *root[2], int dir, vtkCellArray *newPolys, int useOptimization) {
    if (!root[0] || !root[1]) {
        return;
    }

    if (isLeaf(root[0]) && isLeaf(root[1])) {
        return;
    }

    for (int i = 0; i < 4; i++) {
        Octree<T, B> *nodes[2];
        const int c[2] = {faceProcFaceMask[dir][i][0], faceProcFaceMask[dir][i][1]};
        for (int j = 0; j < 2; j++) {
            if (isLeaf(root[j])) {
                nodes[j] = root[j];
            } else {
                nodes[j] = root[j]->children[c[j]];
            }
        }
        contourFaceProc(nodes, faceProcFaceMask[dir][i][2], newPolys, useOptimization);
    }

    const int orders[2][4] =
            {
                    { 0, 0, 1, 1 },
                    { 0, 1, 0, 1 },
            };

    for (int i = 0; i < 4; i++) {
        Octree<T, B> *nodes[4];
        const int c[4] = {faceProcEdgeMask[dir][i][1], faceProcEdgeMask[dir][i][2], faceProcEdgeMask[dir][i][3], faceProcEdgeMask[dir][i][4]};
        const int* order = orders[faceProcEdgeMask[dir][i][0]];
        for (int j = 0; j < 4; j++) {
            if (isLeaf(root[order[j]])) {
                nodes[j] = root[order[j]];
            } else {
                nodes[j] = root[order[j]]->children[c[j]];
            }
        }
        contourEdgeProc(nodes, faceProcEdgeMask[dir][i][5], newPolys, useOptimization);
    }

}

template<class T, class B>
void Octree<T, B>::contourEdgeProc(Octree<T, B> *root[4], int dir, vtkCellArray *newPolys, int useOptimization) {
    if (!root[0] || !root[1] || !root[2] || !root[3]) {
        return;
    }

    if (isLeaf(root[0]) && isLeaf(root[1]) && isLeaf(root[2]) && isLeaf(root[3])) {
        contourProcessEdge(root, dir, newPolys, useOptimization);
    } else {
        for (int i = 0; i < 2; i++) {
            Octree<T, B> *nodes[4];
            const int c[4] = {edgeProcEdgeMask[dir][i][0], edgeProcEdgeMask[dir][i][1], edgeProcEdgeMask[dir][i][2], edgeProcEdgeMask[dir][i][3]};
            for (int j = 0; j < 4; j++) {
                if (isLeaf(root[j])) {
                    nodes[j] = root[j];
                } else {
                    nodes[j] = root[j]->children[c[j]];
                }
            }
            contourEdgeProc(nodes, edgeProcEdgeMask[dir][i][4], newPolys, useOptimization);
        }
    }

}

//namespace std {
//    template <>
//    struct hash<std::pair<unsigned long, unsigned long>> {
//        size_t operator()(const std::pair<unsigned long, unsigned long>& p) const {
//            // 自定义哈希逻辑
//            return std::hash<unsigned long>()(p.first) ^ std::hash<unsigned long>()(p.second);
//        }
//    };
//}
//namespace {
//    std::unordered_map<std::pair<uintptr_t , uintptr_t>, std::unordered_set<OctreeRepresentative*>> edgeMap;
//    std::unordered_set<std::pair<uintptr_t , uintptr_t>> nonManifoldEdges;
//}

template<class T, class B>
void Octree<T, B>::contourProcessEdge(Octree<T, B> *root[4], int dir, vtkCellArray *newPolys, int useOptimization) {
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

        return std::min(std::min(dA, dB), dC);
    };
    for (int i = 0; i < 4; i++) {
        const int edge = processEdgeMask[dir][i];
        int c1 = edges2Vertices[edge][0];
        int c2 = edges2Vertices[edge][1];
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
        std::unordered_set<long long> set;
        for (int i = 0; i < len; i++) {
            if (set.find(tris[i]) != set.end() || tris[i] == -1) {
                return false;
            }
            set.insert(tris[i]);
        }
        return true;
    };

    auto insertTriangle = [&] (long long verticesId[4], const int indices[4], vtkCellArray *newPolys) {
        std::vector<int> tris1(3), tris2(3);
        if (intersections[indices[0]] == 2 && intersections[indices[2]] == 2) {
            tris1 = std::vector<int> {0, 1, 3};
            tris2 = std::vector<int> {1, 2, 3};
        } else {
            tris1 = std::vector<int> {0, 2, 3};
            tris2 = std::vector<int> {0, 1, 2};
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
                newPolys->InsertNextCell(3, tris);
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
                newPolys->InsertNextCell(3, tris);
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
        const double b1_ = std::min(a1_, a2_);
        const double b2_ = std::max(a1_, a2_);
        a1_ = minAngle(v0, v1, v3);
        a2_ = minAngle(v0, v2, v3);
        const double c1_ = std::min(a1_, a2_);
        const double c2_ = std::max(a1_, a2_);
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
                insertTriangle(indices, ins, newPolys);

            } else {
                const int ins[4] = {0, 1, 3, 2};
                insertTriangle(indices, ins, newPolys);
            }
        } else {
            if (flip) {
                const int ins[4] = {2, 3, 1, 0};
                insertTriangle(indices, ins, newPolys);

            } else {
                const int ins[4] = {2, 0, 1, 3};
                insertTriangle(indices, ins, newPolys);
            }
        }
    }
}

template<class T, class B>
void Octree<T, B>::clusterCell(Octree *root, double threshold, int useOptimization) {
    if (!root) {
        return;
    }

    if (isLeaf(root)) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        clusterCell(root->children[i], threshold, useOptimization);
    }

    UnionFind<OctreeRepresentative*> unionFind;
    for (int i = 0; i < 12; i++) {
        Octree<T, B> *nodes[2];
        const int c[2] = {cellProcFaceMask[i][0], cellProcFaceMask[i][1]};
        nodes[0] = root->children[c[0]];
        nodes[1] = root->children[c[1]];
        clusterFace(nodes, cellProcFaceMask[i][2], unionFind, useOptimization);
    }

    Octree<T, B> *n[2][2] = {{nullptr, root}, {root, nullptr}};
    for (int dir = 0; dir < 3; dir++) {
        for (int i = 0; i < 2; i++) {
            Octree<T, B> *nodes[2] = {n[i][0], n[i][1]};
            clusterFace(nodes, dir, unionFind, useOptimization);
        }
    }

    for (int i = 0; i < 6; i++) {
        Octree<T, B> *nodes[4];
        const int c[4] = {cellProcEdgeMask[i][0], cellProcEdgeMask[i][1], cellProcEdgeMask[i][2], cellProcEdgeMask[i][3]};
        for (int j = 0; j < 4; j++) {
            nodes[j] = root->children[c[j]];
        }
        clusterEdge(nodes, cellProcEdgeMask[i][4], unionFind, useOptimization);
    }

    // cluster begin
    // the vertex that are not in the unionFind are need to be added to a one-element new set
    for (int i = 0; i < 8; i++) {
        auto child = root->children[i];
        if (!child) {
            continue;
        }
        if (child->type == OctreeNodeType::Node_Leaf) {
            for (int j = 0; j < 12; j++) {
                if (!child->representative[j]) {
                    continue;
                }
                auto vertex = child->representative[j];
                unionFind.addElement(vertex);
            }
        } else {
            for (auto &vertex : child->clusteredVertex[0]) {
                if (!vertex) {
                    continue;
                }
                unionFind.addElement(vertex);
            }
        }
    }

    auto disjointSets = unionFind.getDisjointSets();
//    bool allCollapsible = true;
    int vertexCnt = 0;

    root->clusteredVertex = new std::vector<OctreeRepresentative*>();
    root->collapsible = true;
    for (auto &it : disjointSets) {
        auto representative = it.first;
        auto set = it.second;
        svd::QefSolver qefSolver;
        glm::vec3 normal(0, 0, 0);
        int edgeIntersection[12];
        int internalIntersection = 0;
        int euler = 0;
        for (int i = 0; i < 12; i++) {
            edgeIntersection[i] = 0;
        }
        bool canMerge = true;
        int mergeNode = 0;
        for (auto &node : set) {
            for (int i = 0; i < 3; i++) {
                int edge = externalEdges[node->cellIndex][i];
                edgeIntersection[edge] += node->edgeIntersection[edge];
            }

            qefSolver.add(node->qef);
            normal += node->averageNormal;
            for (int i = 0; i < 9; i++) {
                int edge = internalEdge[node->cellIndex][i];
                internalIntersection += node->edgeIntersection[edge];
            }
            euler += node->euler;
            if (!node->canMerge) {
                canMerge = false;
            }
            mergeNode++;
        }
        if (!mergeNode) {
            continue;
        }
        svd::Vec3 qefPosition;
        qefSolver.solve(qefPosition, QEF_ERROR, QEF_SWEEPS, QEF_ERROR);
        auto error = qefSolver.getError();
        glm::vec3 position(qefPosition.x, qefPosition.y, qefPosition.z);

        if (!inRegion(position, root->region)) {
            const auto &mp = qefSolver.getMassPoint();
            position = glm::vec3(mp.x, mp.y, mp.z);
            error = qefSolver.getError(mp);
        }

        bool edgeManifold = true;
        for (int f = 0; f < 6; f++) {
            int intersections = 0;
            for (int i = 0; i < 4; i++) {
                intersections += edgeIntersection[h_e_face[f][i]];
            }
            if (intersections != 0 && intersections != 2) {
                edgeManifold = false;
                break;
            }
        }

        OctreeRepresentative *newRepresentative = nullptr;
        cudaMallocManaged(&newRepresentative, sizeof(OctreeRepresentative));
        *newRepresentative = OctreeRepresentative();
        newRepresentative->position = position;
        newRepresentative->qef = qefSolver.getData();
        newRepresentative->averageNormal = glm::normalize(normal);
        newRepresentative->euler = euler - internalIntersection / 4;
        newRepresentative->collapsible = error <= threshold && newRepresentative->euler == 1 && edgeManifold ;
//        newRepresentative->collapsible = error <= threshold && newRepresentative->euler != 0;
//        newRepresentative->collapsible = error <= threshold;
//        newRepresentative->collapsible =  newRepresentative->euler > 0  ;

//        newRepresentative->collapsible = edgeManifold;
        newRepresentative->cellIndex = root->index;
        newRepresentative->internalInters = internalIntersection;
        newRepresentative->canMerge = canMerge;
        newRepresentative->height = root->height;
        if (!newRepresentative->collapsible || !canMerge) {
            root->collapsible = false;
        }
        for (int i = 0; i < 12; i++) {
            newRepresentative->edgeIntersection[i] = edgeIntersection[i];
        }
        for (auto &node : set) {
//            if (!node->canMerge) continue;
            node->parent = newRepresentative;
//            newRepresentative->child.push_back(node);
        }

        newRepresentative = newRepresentative;
        vertexCnt++;
        // only used to free the memory
        root->clusteredVertex->push_back( newRepresentative);
    }

}

template<class T, class B>
void Octree<T,B>::clusterFace(Octree *root[2], int dir, UnionFind<OctreeRepresentative*> & unionFind, int useOptimization) {
    if (!root[0] && !root[1]) {
        return;
    }

    if ( (root[0] == nullptr || isLeaf(root[0])) && (root[1] == nullptr || isLeaf(root[1])) ) {
        return;
    }

    for (int i = 0; i < 4; i++) {
        Octree<T, B> *nodes[2];
        const int c[2] = {faceProcFaceMask[dir][i][0], faceProcFaceMask[dir][i][1]};
        for (int j = 0; j < 2; j++) {
            if (!root[j]) {
                nodes[j] = nullptr;
                continue;
            }
            if (isLeaf(root[j])) {
                nodes[j] = root[j];
            } else {
                nodes[j] = root[j]->children[c[j]];
            }
        }
        clusterFace(nodes, faceProcFaceMask[dir][i][2], unionFind, useOptimization);
    }

    const int orders[2][4] =
            {
                    { 0, 0, 1, 1 },
                    { 0, 1, 0, 1 },
            };

    for (int i = 0; i < 4; i++) {
        Octree<T, B> *nodes[4];
        const int c[4] = {faceProcEdgeMask[dir][i][1], faceProcEdgeMask[dir][i][2], faceProcEdgeMask[dir][i][3], faceProcEdgeMask[dir][i][4]};
        const int* order = orders[faceProcEdgeMask[dir][i][0]];
        for (int j = 0; j < 4; j++) {
            if (!root[order[j]]) {
                nodes[j] = nullptr;
                continue;
            }
            if (isLeaf(root[order[j]])) {
                nodes[j] = root[order[j]];
            }
            else {
                nodes[j] = root[order[j]]->children[c[j]];
            }
        }
        clusterEdge(nodes, faceProcEdgeMask[dir][i][5], unionFind, useOptimization);
    }

}

template<class T, class B>
void Octree<T,B>::clusterEdge(Octree *root[4], int dir, UnionFind<OctreeRepresentative*> & unionFind, int useOptimization) {

    if (!root[0] && !root[1] && !root[2] && !root[3]) {
        return;
    }

    bool canCluster = true;
    for (int i = 0; i < 4; i++) {
        if (root[i] && isInternal(root[i])) {
            canCluster = false;
        }
    }

    if (canCluster) {
        clusterVertex(root, dir, unionFind, useOptimization);
    } else {
        for (int i = 0; i < 2; i++) {
            Octree<T, B> *nodes[4];
            const int c[4] = {edgeProcEdgeMask[dir][i][0], edgeProcEdgeMask[dir][i][1], edgeProcEdgeMask[dir][i][2], edgeProcEdgeMask[dir][i][3]};
            for (int j = 0; j < 4; j++) {
                if (!root[j]) {
                    nodes[j] = nullptr;
                    continue;
                }
                if (isLeaf(root[j])) {
                    nodes[j] = root[j];
                } else {
                    nodes[j] = root[j]->children[c[j]];
                }
            }
            clusterEdge(nodes, edgeProcEdgeMask[dir][i][4], unionFind, useOptimization);
        }
    }
}

template<class T, class B>
void Octree<T,B>::clusterVertex(Octree<T, B> *root[4], int dir, UnionFind<OctreeRepresentative*> &unionFind, int useOptimization) {

    int minIndex = 0;
    long long indices[4] = {-1, -1, -1, -1};
    bool flip = false;
    bool signChange[4] = {false, false, false, false};

    OctreeRepresentative *previous = nullptr;
    for (int i = 0; i < 4; i++) {
        if (!root[i]) {
            continue;
        }
        const int edge = processEdgeMask[dir][i];
        int c1 = edges2Vertices[edge][0];
        int c2 = edges2Vertices[edge][1];
        auto m1 = (root[i]->sign >> c1) & 1;
        auto m2 = (root[i]->sign >> c2) & 1;
        if (m1 ^ m2) {
            auto vertex = root[i]->representative[edge];
            while (vertex->parent) {
                vertex = vertex->parent;
            }
            if (!previous /**&& vertex->canMerge**/ ) {
                previous = vertex;
            }
            unionFind.addElement(vertex);
//            if (vertex->canMerge)
            unionFind.unionSets(previous, vertex);
        }
    }

    for (int i = 0; i < 4; i++) {
        if (!root[i]) {
            continue;
        }
        if (useOptimization == 1) {
            handleAmbiguous(root[i]);
        }
    }
}

template<class T, class B>
void Octree<T, B>::handleAmbiguous(Octree *root) {
    auto sign = root->sign;
    if (!isAmbiguousCase(sign)) {
        // not ambiguous case
        return;
    }
    auto dir = ambiguousFaces[~sign];
    Direction direction = convertToDirection(dir);
    // new tree has not been built completely, so use template tree
    auto neighbor = findNeighbor(root, direction);
    if (!neighbor) {
        return;
    }
    auto neighborSign = neighbor->sign;
    if (isAmbiguousCase(neighborSign)) {
        // two adjacent ambiguous faces can be handled properly
        return;
    }
    std::unordered_set<OctreeRepresentative*> vs;
    auto f = h_e_face[convertToFace(direction)];
    for (int j = 0; j < 4; j++) {
        vs.insert(root->representative[f[j]]);
    }
    if (vs.size() != 1) {
        std::cout << "octree representative error happen" << std::endl;
        return;
    }
    auto ambVertex = *vs.begin();
    auto face = convertToFace(reverseDirection(direction));
    auto edges = h_e_face[face];
    vs.clear();
    for (int j = 0; j < 4; j++) {
        vs.insert(neighbor->representative[edges[j]]);
    }
    if (vs.size() != 2) {
        std::cout << "octree representative error happen!!" << std::endl;
        return;
    }
    auto commonV1 = findCommonAncestor(*vs.begin(), *std::next(vs.begin()));
    if (!commonV1) {
        // not clustering to a vertex
        return;
    }
    auto commonV2 = findCommonAncestor(commonV1, ambVertex);
    while (commonV1 && commonV1 != commonV2) {
        commonV1->collapsible = false;
        commonV1 = commonV1->parent;
    }
}

template<class T, class B>
void Octree<T, B>::destroyOctree(Octree *root) {
    if (!root) {
        return;
    }

    if (!isLeaf(root)) {
        for (int i = 0; i < 8; i++) {
            destroyOctree(root->children[i]);
            root->children[i] = nullptr;
        }
        if (!root->clusteredVertex) {
            return;
        }
        auto clusteredVertex = root->clusteredVertex[0];
        std::unordered_set<OctreeRepresentative *> deleteVertices;
        for (int i = 0; i < clusteredVertex.size(); i++) {
            if (!clusteredVertex[i]) {
                continue;
            }
            // check if the vertex is already deleted
            if (deleteVertices.find(clusteredVertex[i]) == deleteVertices.end()) {
                deleteVertices.insert(clusteredVertex[i]);
//                delete root->clusteredVertex[i];
                cudaFree(clusteredVertex[i]);
            }
            clusteredVertex[i] = nullptr;
        }
        delete root->clusteredVertex;
    }

}
