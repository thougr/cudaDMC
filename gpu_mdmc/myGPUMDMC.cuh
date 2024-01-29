//
// Created by hougr t on 2024/1/28.
//

#ifndef CUDADMC_MYGPUMDMC_CUH
#define CUDADMC_MYGPUMDMC_CUH


#include "../isosurfacesAlgorithm.h"

class myGPUMDMC : public isosurfacesAlgorithm {
public:
    static myGPUMDMC* New();
vtkTypeMacro(myGPUMDMC, isosurfacesAlgorithm);
    void process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                 vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                 vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) override;
    bool supportAdaptiveMeshing() override;
};


#endif //CUDADMC_MYGPUMDMC_CUH
