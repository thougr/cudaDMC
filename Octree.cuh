//
// Created by hougr t on 2023/11/17.
//
#pragma once
#ifndef MC_OCTREE_H
#define MC_OCTREE_H

//#include <vtkLocator.h>
//#include <vtkIncrementalPointLocator.h>
//#include <vtkCellArray.h>
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "qef.cuh"
#include <vector>
#include <vtkDataArray.h>
#include <vtkIncrementalPointLocator.h>
#include <vtkCellArray.h>
//#include "UnionFind.h"
#include "common.cuh"
#include <stdgpu/functional.h>

template <typename T>
class UnionFind;

template <class T, class B>
class Octree;

enum OctreeNodeType {
    Node_None,
    Node_Internal,
    Node_Psuedo,
    Node_Leaf,
};


struct Region {
    glm::u32vec3 origin;
    glm::u32vec3 size; // cubes size
    uint64_t voxelsCnt; // cubes count
    glm::u32vec3 conceptualSize; // round to pow2 used for branch-on-need octree
};

template <typename T>
struct VoxelsData {
    T *scalars;
    glm::u32vec3 cubeDims; // cubes dimensions. not vertices but cubes!!!
    glm::u32vec3 dims; // vertices dimensions
    glm::u32vec3 conceptualDims; // round to pow2 used for branch-on-need octree
};


struct OctreeRepresentative {
    long long index;
    glm::vec3 position;
    glm::vec3 averageNormal;
    svd::QefData qef;

    // used for manifold
    OctreeRepresentative *parent;
    bool collapsible;
    int cellIndex;
    // intersection count for every edge
    int edgeIntersection[12];
    int euler;

    bool canMerge;
    int height;

    int layerId;

    // only for debug
    int sign;
//    std::vector<OctreeRepresentative *> child;
    int internalInters;

    __host__ __device__ OctreeRepresentative() {
        index = -1;
        position = glm::vec3(0, 0, 0);
        averageNormal = glm::vec3(0, 0, 0);
        qef = svd::QefData();
        parent = nullptr;
        collapsible = true;
        cellIndex = -1;
        sign = -1;
        canMerge = true;
        height = 0;
        layerId = -1;
        for (int i = 0; i < 12; i++) {
            edgeIntersection[i] = 0;
        }
        euler = 1;
    }

};

template <class T, class B>
class Octree {
public:
    Octree* parent;
    Octree* children[8];
    Octree* neighbour[6];
    int depth;
    int height;
    int index;
    int dims[3];
    T minScalar;
    T maxScalar;
    Region region;
    T scalar[8];
    glm::vec3 normal[8];
    uint8_t sign;
    OctreeRepresentative *representative[12];

    std::vector<OctreeRepresentative*> *clusteredVertex;
    OctreeRepresentative *representativeBegin;
    int clusteredVertexCnt;

    // accumulated representative count
//    int representativeCnt;
    double isoValue;
    OctreeNodeType type;
    bool collapsible;
    // index at current layer
    int idLayer;
//    bool canMerge;

public:
    __host__ __device__ void init() {
        parent = nullptr;
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
        }
        depth = 0;
        index = 0;
        dims[0] = 0;
        dims[1] = 0;
        dims[2] = 0;
        sign = 0;
        this->collapsible = false;
//        representativeCnt = 0;
        idLayer = -1;
        representativeBegin = nullptr;
        clusteredVertexCnt = 0;
//        this->canMerge = true;
//        representative.resize(12);
        for (int i = 0; i < 12; i++) {
            representative[i] = nullptr;
        }
        clusteredVertex = nullptr;

    }
    __host__ __device__ Octree() {
        parent = nullptr;
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
        }
        depth = 0;
        index = 0;
        dims[0] = 0;
        dims[1] = 0;
        dims[2] = 0;
        sign = 0;
        this->collapsible = false;
//        representativeCnt = 0;
        idLayer = -1;
        representativeBegin = nullptr;
        clusteredVertexCnt = 0;
//        this->canMerge = true;
//        representative.resize(12);
        for (int i = 0; i < 12; i++) {
            representative[i] = nullptr;
        }
        clusteredVertex = nullptr;
    }
    __host__ __device__ Octree(int depth, int index, Region region, Octree* parent, OctreeNodeType type) {
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
        }
        this->depth = depth;
        this->index = index;
        this->region = region;
        this->parent = parent;
        this->type = type;
        this->sign = 0;
        this->collapsible = false;
//        representativeCnt = 0;
        idLayer = -1;
        representativeBegin = nullptr;
        clusteredVertexCnt = 0;
//        this->canMerge = true;
//        representative.resize(12);
        for (int i = 0; i < 12; i++) {
            representative[i] = nullptr;
        }
        clusteredVertex = nullptr;
    }

    __host__ __device__ Octree(int depth, int height, int index, Region region, Octree* parent, OctreeNodeType type) {
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
        }
        this->depth = depth;
        this->index = index;
        this->region = region;
        this->parent = parent;
        this->height = height;
        this->type = type;
        this->sign = 0;
        this->collapsible = false;
//        representativeCnt = 0;
        idLayer = -1;
        representativeBegin = nullptr;
        clusteredVertexCnt = 0;
//        this->canMerge = true;
//        representative.resize(12);
        for (int i = 0; i < 12; i++) {
            representative[i] = nullptr;
        }
        clusteredVertex = nullptr;
    }


    __host__ __device__ static bool isLeaf(const Region &region) {
        return region.voxelsCnt == 1;
    }

    __host__ __device__ static bool isLeaf(const Octree *root) {
        return isLeaf(root->type);
    }

    __host__ __device__ static bool isLeaf(OctreeNodeType type) {
        return type == Node_Leaf || type == Node_Psuedo;
    }

    __host__ __device__ static bool isInternal(const Octree *root) {
        return isInternal(root->type);
    }

    __host__ __device__ static bool isInternal(OctreeNodeType type) {
        return type == Node_Internal;
    }

    __device__ static bool calculateMDCRepresentative(Octree *root, Octree *templateRoot, OctreeRepresentative *allVertex, int *globalIndex, double isovalue, int useOptimization=0);

    __device__ static bool invertSign(Octree *root, Octree *templateRoot, double isovalue);
    __device__ static void calculateAccurateRepresentative(Octree *root, OctreeRepresentative *allVertex, int *globalIndex, double isovalue, uint8_t sign, int clusterDisable);
    __host__ __device__ static Octree<T,B>* findNeighbor(Octree *root, Direction dir);

    __host__ static void generateVerticesIndices(Octree *root,  vtkIncrementalPointLocator *locator, vtkDataArray* newScalars);
    __host__ static void contourCellProc(Octree *root, vtkCellArray* newPolys, int useOptimization=0);
    __host__ static void contourFaceProc(Octree *root[2], int dir, vtkCellArray* newPolys, int useOptimization=0);
    __host__ static void contourEdgeProc(Octree *root[4], int dir, vtkCellArray* newPolys, int useOptimization=0);
    __host__ static void contourProcessEdge(Octree *root[4], int dir, vtkCellArray* newPolys, int useOptimization=0);


    __host__ static void clusterCell(Octree *root, double threshold, int useOptimization=0);
    __host__ static void clusterFace(Octree *root[2], int dir, UnionFind<OctreeRepresentative*> &unionFind, int useOptimization=0);
    __host__ static void clusterEdge(Octree *root[4], int dir, UnionFind<OctreeRepresentative*> &unionFind, int useOptimization=0);
    __host__ static void clusterVertex(Octree *root[4], int dir, UnionFind<OctreeRepresentative*> &unionFind, int useOptimization=0);
    __host__ static void handleAmbiguous(Octree *root);
    __host__ static void destroyOctree(Octree *root);

};

#include "Octree.inl"

#endif //MC_OCTREE_H
