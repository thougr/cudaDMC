//
// Created by hougr t on 2024/2/2.
//

#ifndef CUDADMC_GPUMC_CUH
#define CUDADMC_GPUMC_CUH

#include "../isosurfacesAlgorithm.h"
class gpuMC : public isosurfacesAlgorithm {
public:
    static gpuMC* New();
    vtkTypeMacro(gpuMC, isosurfacesAlgorithm);
    void process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                 vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                 vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) override;
};


#endif //CUDADMC_GPUMC_CUH
