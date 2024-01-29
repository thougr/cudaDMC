//
// Created by hougr t on 2024/1/29.
//
#include "table.cuh"

__device__ __managed__ glm::vec3 *ambiguousFaces;
void ambiguousFacesInit() {
    cudaMallocManaged(&ambiguousFaces, 256 * sizeof(glm::vec3));
    ambiguousFaces[VERTICES0 | VERTICES3] = {0.0, 0.0, -1.0};ambiguousFaces[VERTICES0 | VERTICES3 | VERTICES4] = {0.0, 0.0, -1.0};ambiguousFaces[VERTICES0 | VERTICES3 | VERTICES7] = {0.0, 0.0, -1.0};
    ambiguousFaces[VERTICES1 | VERTICES2] = {0.0, 0.0, -1.0};ambiguousFaces[VERTICES1 | VERTICES2 | VERTICES5] = {0.0, 0.0, -1.0};ambiguousFaces[VERTICES1 | VERTICES2 | VERTICES6] = {0.0, 0.0, -1.0};
    ambiguousFaces[VERTICES4 | VERTICES7] = {0.0, 0.0, 1.0};ambiguousFaces[VERTICES4 | VERTICES7 | VERTICES0] = {0.0, 0.0, 1.0};ambiguousFaces[VERTICES4 | VERTICES7 | VERTICES3] = {0.0, 0.0, 1.0};
    ambiguousFaces[VERTICES5 | VERTICES6] = {0.0, 0.0, 1.0};ambiguousFaces[VERTICES5 | VERTICES6 | VERTICES1] = {0.0, 0.0, 1.0};ambiguousFaces[VERTICES5 | VERTICES6 | VERTICES2] = {0.0, 0.0, 1.0};
    ambiguousFaces[VERTICES0 | VERTICES5] = {0.0, -1.0, 0.0};ambiguousFaces[VERTICES0 | VERTICES5 | VERTICES7] = {0.0, -1.0, 0.0};ambiguousFaces[VERTICES0 | VERTICES5 | VERTICES2] = {0.0, -1.0, 0.0};
    ambiguousFaces[VERTICES1 | VERTICES4] = {0.0, -1.0, 0.0};ambiguousFaces[VERTICES1 | VERTICES4 | VERTICES6] = {0.0, -1.0, 0.0};ambiguousFaces[VERTICES1 | VERTICES4 | VERTICES3] = {0.0, -1.0, 0.0};
    ambiguousFaces[VERTICES2 | VERTICES7] = {0.0, 1.0, 0.0};ambiguousFaces[VERTICES2 | VERTICES7 | VERTICES5] = {0.0, 1.0, 0.0};ambiguousFaces[VERTICES2 | VERTICES7 | VERTICES0] = {0.0, 1.0, 0.0};
    ambiguousFaces[VERTICES3 | VERTICES6] = {0.0, 1.0, 0.0};ambiguousFaces[VERTICES3 | VERTICES6 | VERTICES4] = {0.0, 1.0, 0.0};ambiguousFaces[VERTICES3 | VERTICES6 | VERTICES1] = {0.0, 1.0, 0.0};
    ambiguousFaces[VERTICES0 | VERTICES6] = {-1.0, 0.0, 0.0};ambiguousFaces[VERTICES0 | VERTICES6 | VERTICES1] = {-1.0, 0.0, 0.0};ambiguousFaces[VERTICES0 | VERTICES6 | VERTICES7] = {-1.0, 0.0, 0.0};
    ambiguousFaces[VERTICES4 | VERTICES2] = {-1.0, 0.0, 0.0};ambiguousFaces[VERTICES4 | VERTICES2 | VERTICES3] = {-1.0, 0.0, 0.0};ambiguousFaces[VERTICES4 | VERTICES2 | VERTICES5] = {-1.0, 0.0, 0.0};
    ambiguousFaces[VERTICES1 | VERTICES7] = {1.0, 0.0, 0.0};ambiguousFaces[VERTICES1 | VERTICES7 | VERTICES0] = {1.0, 0.0, 0.0};ambiguousFaces[VERTICES1 | VERTICES7 | VERTICES6] = {1.0, 0.0, 0.0};
    ambiguousFaces[VERTICES3 | VERTICES5] = {1.0, 0.0, 0.0};ambiguousFaces[VERTICES3 | VERTICES5 | VERTICES4] = {1.0, 0.0, 0.0};ambiguousFaces[VERTICES3 | VERTICES5 | VERTICES2] = {1.0, 0.0, 0.0};
}

__device__ __managed__ bool *ambiguousCasesComplement;
void ambiguousCasesComplementInit() {
    cudaMallocManaged(&ambiguousCasesComplement, 256 * sizeof(bool));
    for (int i = 0; i < 256; ++i) {
        ambiguousCasesComplement[i] = false;
    }
    for (auto i : ambiguousCases) {
        ambiguousCasesComplement[~i] = true;
    }
}
