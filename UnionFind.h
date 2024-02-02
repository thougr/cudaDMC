//
// Created by hougr t on 2023/11/30.
//

#ifndef MC_UNIONFIND_H
#define MC_UNIONFIND_H
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// 并查集数据结构
template <typename T>
class UnionFind {
private:
    std::unordered_map<T, T> parent;
    std::unordered_map<T, int> rank;

public:
    UnionFind() {}

    // 查找元素所属的集合（根节点）
    T find(T x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // 路径压缩
        }
        return parent[x];
    }

    // 合并两个集合
    void unionSets(T x, T y) {
        T rootX = find(x);
        T rootY = find(y);

        if (rootX != rootY) {
            // 按秩合并
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootX] = rootY;
                rank[rootY]++;
            }
        }
    }

    // 添加元素到并查集
    void addElement(T x) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            rank[x] = 0;
        }
    }

    // get different sets and their elements
    std::unordered_map<T, std::vector<T>> getDisjointSets() {
        std::unordered_map<T, std::vector<T>> sets;
        for (auto &p : parent) {
            sets[find(p.first)].push_back(p.first);
        }
        return sets;
    }
};

#endif //MC_UNIONFIND_H
