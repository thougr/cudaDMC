//
// Created by hougr t on 2023/11/30.
//

#ifndef MC_UNIONFIND_H
#define MC_UNIONFIND_H

#include <stdgpu/unordered_map.cuh>
#include <thrust/device_vector.h>
// 并查集数据结构
class UnionFind {

public:
    stdgpu::unordered_map<int, int> parent;
    stdgpu::unordered_map<int, int> rank;
//    thrust::device_vector<int> result;
//    stdgpu::vector<int> result;
//    stdgpu::vector<int> result_parent;
    int *result;
    int *result_parent;
    int *lock;
    int *count;

    // 查找元素所属的集合（根节点）
    __device__ int find(int x) {
        auto it = parent.find(x);
        if (it->second != x) {
            parent.insert({x, find(it->second)}); // 路径压缩
        }
        return parent.find(x)->second;
    }

public:
    __host__ __device__ UnionFind(int n) {
        if (n > 0) {
            parent = stdgpu::unordered_map<int, int>::createDeviceObject(n);
            rank = stdgpu::unordered_map<int, int>::createDeviceObject(n);
//            result = stdgpu::vector<int>::createDeviceObject(n);
//            result_parent = stdgpu::vector<int>::createDeviceObject(n);
//            result = thrust::device_vector<int>(n);
            cudaMallocManaged(&result, n * sizeof(int));
            cudaMallocManaged(&result_parent, n * sizeof(int));

            cudaMallocManaged(&count, sizeof(int));
            cudaMallocManaged(&lock, sizeof(int));
            cudaMemset(count, n, sizeof(int));
            cudaMemset(lock, 0, sizeof(int));

        }
    }

    __host__ __device__ ~UnionFind() {
        stdgpu::unordered_map<int, int>::destroyDeviceObject(parent);
        stdgpu::unordered_map<int, int>::destroyDeviceObject(rank);
//        stdgpu::vector<int>::destroyDeviceObject(result);
//        stdgpu::vector<int>::destroyDeviceObject(result_parent);
        cudaFree(result);
        cudaFree(result_parent);
        cudaFree(count);
        cudaFree(lock);
    }

    // 合并两个集合
    __device__ void unionSets(int x, int y) {
        while(atomicCAS(lock, 0, 1) != 0);
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            // 按秩合并
            auto rankX = rank.find(rootX)->second;
            auto rankY = rank.find(rootY)->second;
            if (rankX < rankY) {
                parent.insert({rootX, rootY});
            } else if (rankX > rankY) {
                parent.insert({rootY, rootX});
            } else {
                parent.insert({rootX, rootY});
                rank.insert({rootY, rankY + 1});
            }
        }
        atomicExch(lock, 0);
    }

    // https://stackoverflow.com/questions/31194291/cuda-mutex-why-deadlock%5B/url%5D
    __device__ void addElement(int x) {
        while(atomicCAS(lock, 0, 1) != 0);
//        bool blocked = true;
//        while (blocked) {
//            if (atomicCAS(lock, 0, 1) == 0) {
////        if (parent.find(x) == parent.end()) {
////            parent.insert({x, x});
//
////            parent.emplace(x, x);
////            rank.insert({x, 0});
////            result[parent.size() - 1] = x;
////        }
//                atomicExch(lock, 0);
//                blocked = false;
//
//            }
//
//        }

        atomicExch(lock, 0);
    }
    // get different sets and their elements
//    __host__ __device__ stdgpu::unordered_map<T, stdgpu::vector<T>> getDisjointSets() {
//        stdgpu::unordered_map<T, stdgpu::vector<T>> sets;
//        for (auto &p : parent) {
//            sets[find(p.first)].push_back(p.first);
//        }
//        return sets;
//    }
    // CUDA内核：找到每个元素的根
//    __global__ static void findRoots(UnionFind* uf, int* elements, int* roots, int n) {
//        unsigned stride = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//        unsigned blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
//        unsigned offset = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//        for (int i = offset; i < n; i += stride) {
//            roots[i] = uf->find(elements[i]);
//        }
//    }

    __device__ void getDisjointSets() {
        // sort the result int device
//        auto range_vec = result.device_range();
//        thrust::sort(result.device_begin(), result.device_end(), [=] __device__(int a, int b) {
//            return find(a) < find(b);
//        });
        for (int i = 0; i < *count; i++) {
            result_parent[i] = find(result[i]);
        }

//        std::unordered_map<int, std::vector<int>> sets;
//        for (int i = 0; i < *count; i++) {
//            sets[roots[i]].push_back(i);
//        }
//        return sets;
    }
};

#endif //MC_UNIONFIND_H
