//
// Created by hougr t on 2023/12/11.
//

#ifndef MC_COMMON_H
#define MC_COMMON_H
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "table.cuh"

namespace {
    glm::u32vec3 findLargerClosestPower2Vector(const glm::u32vec3 &dims, int &height) {
        auto x = 1 << (32 - __builtin_clz(dims.x - 1));
        auto y = 1 << (32 - __builtin_clz(dims.y - 1));
        auto z = 1 << (32 - __builtin_clz(dims.z - 1));
        // find the max of x,y,z
        auto max = x;
        if (y > max) {
            max = y;
        }
        if (z > max) {
            max = z;
        }
        // find the position 1 of max
        height = 32 - __builtin_clz(max);
        return {max, max, max};
    }
// 类型萃取，用于确定 sum 的类型
    template <typename T>
    struct SumType { using type = double; };

// 对于 glm::vec3 类型，特化 sum 的类型
    template <>
    struct SumType<glm::vec3> { using type = glm::vec3; };

    // 提供 zero_value 的实现，用于不同类型的零值初始化
    template <typename T>
    T zero_value() {
        return T(0);
    }

// 对 glm::vec3 类型的特化
    template <>
    glm::vec3 zero_value<glm::vec3>() {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    template <class T>
    typename SumType<T>::type trilinearInterpolation(const std::vector<T>& scalars, const glm::vec3& pos) {
        typename SumType<T>::type sum = zero_value<typename SumType<T>::type>();
        for (int i = 0; i < 8; i++) {
            auto v = localVerticesPos[i];
            sum += scalars[i] * (1.f - fabs(v.x - pos.x)) * (1.f - fabs(v.y - pos.y)) * (1.f - fabs(v.z - pos.z));
        }
        return sum;
    }

    template <class T>
    typename SumType<T>::type trilinearInterpolation(const T scalars[8], const glm::vec3& pos) {
        typename SumType<T>::type sum = zero_value<typename SumType<T>::type>();
        for (int i = 0; i < 8; i++) {
            auto v = localVerticesPos[i];
            sum += scalars[i] * (1.f - fabs(v.x - pos.x)) * (1.f - fabs(v.y - pos.y)) * (1.f - fabs(v.z - pos.z));
        }
        return sum;
    }

    template <class T>
    T trilinearInterpolation(const std::vector<T> &scalars, const std::vector<glm::vec3> &vertexPos, const glm::vec3 &pos) {
//        T sum = 0;
        T sum = zero_value<T>();
        for (int i = 0; i < vertexPos.size(); i++) {
            auto v = vertexPos[i];
            sum += scalars[i] * (float)((1.f - fabs(v.x - pos.x)) * (1.f - fabs(v.y - pos.y)) * (1.f - fabs(v.z - pos.z)));
        }
        return sum;
    }

//
//    // check if C is on the line AB
//    bool isPointOnLine(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C) {
//        // 计算向量AB和向量AC
//        glm::vec3 AB = B - A;
//        glm::vec3 AC = C - A;
//
//        // 计算叉积
//        glm::vec3 crossProduct = glm::cross(AB, AC);
//
//        // 如果叉积的长度接近0，则C在直线上
//        return glm::length(crossProduct) < 1e-6; // 1e-6 是一个足够小的数来处理浮点数的精度问题
//    }
//
//    bool isPointOnPlane(const glm::vec3& P, const glm::vec3& A, const glm::vec3& normal) {
//        float distance = glm::dot(normal, P - A);
//        return std::abs(distance) < 1e-6;  // 处理浮点数精度问题
//    }
//
//// 假设 A, B, C, D 是长方形的四个顶点
//// P 是我们要检查的新顶点
//    bool checkIfPointOnRectanglePlane(const glm::vec3& P, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C) {
//        glm::vec3 AB = B - A;
//        glm::vec3 AC = C - A;
//        glm::vec3 normal = glm::cross(AB, AC);  // 平面法线
//        return isPointOnPlane(P, A, normal);
//    }
    __host__ __device__ auto position2Index(glm::u32vec3 pos, glm::u32vec3 dims) {
        return pos.x + pos.y * dims[0] + pos.z * dims[0] * dims[1];
    };
}

#endif //MC_COMMON_H
