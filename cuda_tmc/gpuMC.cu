//
// Created by hougr t on 2024/2/2.
//

#include "gpuMC.cuh"

#include "vtkArrayDispatch.h"
#include "vtkCellArray.h"
#include "vtkCharArray.h"
#include "vtkDataArrayRange.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkImageTransform.h"
#include "vtkIncrementalPointLocator.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkIntArray.h"
#include "vtkLongArray.h"
#include "vtkMarchingCubesTriangleCases.h"
#include "vtkMath.h"
#include "vtkMergePoints.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkShortArray.h"
#include "vtkStreamingDemandDrivenPipeline.h"


vtkStandardNewMacro(gpuMC);
#define GLM_FORCE_CUDA
#include "../glm/glm.hpp"
#include "MarchingCubes.h"


namespace {

    struct ComputeGradientWorker
    {
        template <class ScalarArrayT>
        __host__ void operator()(ScalarArrayT* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                                 vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                                 vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) const
        {
            int extent[6];
            double value = values[0];
            vtkTypeBool ComputeNormals = newNormals != nullptr;
            vtkTypeBool ComputeGradients = newGradients != nullptr;
            vtkTypeBool ComputeScalars = newScalars != nullptr;

            vtkInformation* inInfo = self->GetExecutive()->GetInputInformation(0, 0);
            inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent);
            p_mc::MarchingCubes mc;
            int nr_v;
            float* vertices;
            float* normals;
            int nr_t;
            int* triangles;

            auto begin = std::chrono::high_resolution_clock::now();;
            mc.mc_sharedvertex(value, dims, scalarsArray, nr_v, &vertices, &normals, nr_t, &triangles);

            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - begin;
                double seconds = duration.count();
                std::cout << "总时间: " << seconds << " 秒" << std::endl;
            }

            std::vector<vtkIdType> globalId;
            for (int i = 0; i < nr_v; i++) {
                double x[3], n[3];
                x[0] = vertices[i * 3];
                x[1] = vertices[i * 3 + 1];
                x[2] = vertices[i * 3 + 2];
                n[0] = normals[i * 3];
                n[1] = normals[i * 3 + 1];
                n[2] = normals[i * 3 + 2];
                vtkIdType ptId;
                if (locator->InsertUniquePoint(x, ptId))
                {
                    if (ComputeScalars)
                    {
                        newScalars->InsertTuple(ptId, &value);
                    }
                    if (ComputeGradients)
                    {
                        newGradients->InsertTuple(ptId, n);
                    }
                    if (ComputeNormals)
                    {
                        vtkMath::Normalize(n);
                        newNormals->InsertTuple(ptId, n);
                    }
                }
                globalId.push_back(ptId);
            }
            for (int i = 0; i < nr_t; i++) {
                vtkIdType ptIds[3];
                ptIds[0] = globalId[triangles[i * 3]];
                ptIds[1] = globalId[triangles[i * 3 + 1]];
                ptIds[2] = globalId[triangles[i * 3 + 2]];
                if (ptIds[0] != ptIds[1] && ptIds[0] != ptIds[2] && ptIds[1] != ptIds[2])
                {
                    newPolys->InsertNextCell(3, ptIds);
                }
            }

        }
    };



}

void gpuMC::process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                        vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                        vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) {

    using Dispatcher = vtkArrayDispatch::Dispatch;
    ComputeGradientWorker worker;
    if (!Dispatcher::Execute(scalarsArray, worker, this, dims, this->Locator, newScalars, newGradients,
                             newNormals, newPolys, values, numValues)) { // Fallback to slow path for unknown arrays:
        std::cout << "Fallback to slow path for unknown arrays" << std::endl;
        worker(scalarsArray, this, dims, this->Locator, newScalars, newGradients, newNormals, newPolys,
               values, numValues);
    }
}
