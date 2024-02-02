//
// Created by hougr t on 2024/2/2.
//

#include <chrono>
#include "gpuDMC.cuh"

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

vtkStandardNewMacro(gpuDMC);
#include "../UniformGrid.h"
#include "../DualMarchingCubes.h"


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

            using SurfaceCase = p_mc::UniformGrid::SurfaceCase;

            std::map<std::string,int> config;
            config["valence"] = 0; // computes vertex valences
            config["element-quality"] = 0; // computes element quality
//            config["p3X3YColor"] = 1; // color based mesh simplification
//            config["p3X3YOld"] = 1;  // simplify isolated elements, i.e. no neighbor with same valence pattern
//            config["p3333"] = 1; // simplify vertex valence pattern 3333
            config["p3X3YColor"] = 0; // color based mesh simplification
            config["p3X3YOld"] = 0;  // simplify isolated elements, i.e. no neighbor with same valence pattern
            config["p3333"] = 0; // simplify vertex valence pattern 3333
            config["halfedge-datastructure"] = 0; // computes halfedge data structure for quad only mesh
            config["non-manifold"] = 0; // compute number of non-manifold edges
//            std::array<int, 3> dim{64,64,64};
            p_mc::DualMarchingCubes dmc;

            // Init: generate a synthetic data set
            std::cout << " ... init" << std::endl;
//            dmc.init(dim,SurfaceCase::GenusTwo);
            dmc.init(dims, scalarsArray);
            const float i0 = value;

            // mesh
            using Vertex = p_mc::DualMarchingCubes::Vertex;
            using Normal = p_mc::DualMarchingCubes::Normal;
            using Halfedge = p_mc::DualMarchingCubes::Halfedge;
            using HalfedgeFace = p_mc::DualMarchingCubes::HalfedgeFace;
            using HalfedgeVertex = p_mc::DualMarchingCubes::HalfedgeVertex;
            using Triangle = p_mc::DualMarchingCubes::Triangle;
            using Quadrilateral = p_mc::DualMarchingCubes::Quadrilateral;

            std::vector<Vertex> v;
            std::vector<Normal> n;
            std::vector<Triangle> t;
            std::vector<Quadrilateral> q;
            std::vector<Halfedge> h;
            std::vector<HalfedgeFace> hf;
            std::vector<HalfedgeVertex> hv;

            // compute iso-surface
            std::cout << " ... compute iso-surface" << std::endl;
            auto begin = std::chrono::high_resolution_clock::now();;

            dmc.dualMC(i0, v, n, t, q, h, hf, hv, config);

            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - begin;
                double seconds = duration.count();
                std::cout << "总时间: " << seconds << " 秒" << std::endl;
            }

            std::vector<vtkIdType> globalId;
            const int nr_v = v.size();
            const int nr_t = t.size();
            for (int i = 0; i < nr_v; i++) {
                auto &point = v[i];
                auto &normal = n[i];
                double x[3], n1[3];
                x[0] = point[0];
                x[1] = point[1];
                x[2] = point[2];
                n1[0] = normal[0];
                n1[1] = normal[1];
                n1[2] = normal[2];

                vtkIdType ptId;
                if (locator->InsertUniquePoint(x, ptId))
                {
                    if (ComputeScalars)
                    {
                        newScalars->InsertTuple(ptId, &value);
                    }
                    if (ComputeGradients)
                    {
                        newGradients->InsertTuple(ptId, n1);
                    }
                    if (ComputeNormals)
                    {
                        vtkMath::Normalize(n1);
                        newNormals->InsertTuple(ptId, n1);
                    }
                }
                globalId.push_back(ptId);
            }
            for (int i = 0; i < nr_t; i++) {
                vtkIdType ptIds[3];
                ptIds[0] = globalId[t[i][0]];
                ptIds[1] = globalId[t[i][1]];
                ptIds[2] = globalId[t[i][2]];
                if (ptIds[0] != ptIds[1] && ptIds[0] != ptIds[2] && ptIds[1] != ptIds[2])
                {
                    newPolys->InsertNextCell(3, ptIds);
                }
            }

        }
    };



}

void gpuDMC::process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
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
