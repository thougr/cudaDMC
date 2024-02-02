//
// Created by hougr t on 2024/2/2.
//

#ifndef CUDADMC_GPUDMC_CUH
#define CUDADMC_GPUDMC_CUH

#include "../../isosurfacesAlgorithm.h"

class gpuDMC : public isosurfacesAlgorithm {
public:
    static gpuDMC* New();
    vtkTypeMacro(gpuDMC, isosurfacesAlgorithm);
    void process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                 vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                 vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) override;
};


#endif //CUDADMC_GPUDMC_CUH
