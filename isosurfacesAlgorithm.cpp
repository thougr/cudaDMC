//
// Created by hougr t on 2023/10/27.
//

#include <vtkRemoveDuplicatePolys.h>
#include "isosurfacesAlgorithm.h"

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
#include "vtkStructuredPoints.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkUnsignedShortArray.h"
#include <chrono>
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "common.cuh"
//VTK_ABI_NAMESPACE_BEGIN
//vtkStandardNewMacro(isosurfacesAlgorithm);
//#define CAL_TOPOLOGY_ERROR

//namespace {
//template <class T>
//}

isosurfacesAlgorithm::isosurfacesAlgorithm()
{
    this->ContourValues = vtkContourValues::New();
    this->ComputeNormals = 1;
    this->ComputeGradients = 0;
    this->ComputeScalars = 1;
    this->Locator = nullptr;
    this->AdaptiveThreshold = 0;
}

isosurfacesAlgorithm::~isosurfacesAlgorithm()
{
    this->ContourValues->Delete();
    if (this->Locator)
    {
        this->Locator->UnRegister(this);
        this->Locator = nullptr;
    }
}


vtkMTimeType isosurfacesAlgorithm::GetMTime()
{
    vtkMTimeType mTime = this->Superclass::GetMTime();
    vtkMTimeType mTime2 = this->ContourValues->GetMTime();

    mTime = (mTime2 > mTime ? mTime2 : mTime);
    if (this->Locator)
    {
        mTime2 = this->Locator->GetMTime();
        mTime = (mTime2 > mTime ? mTime2 : mTime);
    }

    return mTime;
}

//namespace {
//    template <class ScalarRangeT>
//    void vtkMarchingCubesComputePointGradient(
//            int i, int j, int k, const ScalarRangeT s, int dims[3], vtkIdType sliceSize, double n[3])
//    {
//        double sp, sm;
//
//        // x-direction
//        if (i == 0)
//        {
//            sp = s[i + 1 + j * dims[0] + k * sliceSize];
//            sm = s[i + j * dims[0] + k * sliceSize];
//            n[0] = sm - sp;
//        }
//        else if (i == (dims[0] - 1))
//        {
//            sp = s[i + j * dims[0] + k * sliceSize];
//            sm = s[i - 1 + j * dims[0] + k * sliceSize];
//            n[0] = sm - sp;
//        }
//        else
//        {
//            sp = s[i + 1 + j * dims[0] + k * sliceSize];
//            sm = s[i - 1 + j * dims[0] + k * sliceSize];
//            n[0] = 0.5 * (sm - sp);
//        }
//
//        // y-direction
//        if (j == 0)
//        {
//            sp = s[i + (j + 1) * dims[0] + k * sliceSize];
//            sm = s[i + j * dims[0] + k * sliceSize];
//            n[1] = sm - sp;
//        }
//        else if (j == (dims[1] - 1))
//        {
//            sp = s[i + j * dims[0] + k * sliceSize];
//            sm = s[i + (j - 1) * dims[0] + k * sliceSize];
//            n[1] = sm - sp;
//        }
//        else
//        {
//            sp = s[i + (j + 1) * dims[0] + k * sliceSize];
//            sm = s[i + (j - 1) * dims[0] + k * sliceSize];
//            n[1] = 0.5 * (sm - sp);
//        }
//
//        // z-direction
//        if (k == 0)
//        {
//            sp = s[i + j * dims[0] + (k + 1) * sliceSize];
//            sm = s[i + j * dims[0] + k * sliceSize];
//            n[2] = sm - sp;
//        }
//        else if (k == (dims[2] - 1))
//        {
//            sp = s[i + j * dims[0] + k * sliceSize];
//            sm = s[i + j * dims[0] + (k - 1) * sliceSize];
//            n[2] = sm - sp;
//        }
//        else
//        {
//            sp = s[i + j * dims[0] + (k + 1) * sliceSize];
//            sm = s[i + j * dims[0] + (k - 1) * sliceSize];
//            n[2] = 0.5 * (sm - sp);
//        }
//    }
//
//    struct ComputeGradientWorker
//    {
//        template <class ScalarArrayT>
//        void operator()(ScalarArrayT* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
//                        vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
//                        vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) const
//        {
//            std::cout << "ComputeGradientWorker" << clock() << std::endl;
//            clock_t start = clock(), end;
//            int extent[6];
//            double value = values[0];
//
//            vtkInformation* inInfo = self->GetExecutive()->GetInputInformation(0, 0);
//            inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent);
//            const auto scalars = vtk::DataArrayValueRange<1>(scalarsArray);
//            int flag;
//            int sliceSize = dims[0] * dims[1];
//            int edges2Vertices[12][2] = { { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 }, { 4, 5 }, { 5, 6 },
//                                          { 7, 6 }, { 4, 7 }, { 0, 4 }, { 1, 5 }, { 3, 7 }, { 2, 6 } };
//            // 一个cube的标号->scalars下标偏移,决定标号在cube所在的位置
//            int localVerticesIndex[8] = {0, 1, dims[0] + 1, dims[0], dims[0] * dims[1], dims[0] * dims[1] + 1,
//                                         dims[0] * dims[1] + dims[0] + 1, dims[0] * dims[1] + dims[0]};
//            int localVerticesPos[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
//                                          {0, 0, 1},{1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
//            vtkMarchingCubesTriangleCases* cases = vtkMarchingCubesTriangleCases::GetCases();
//            auto localVertices2GlobalIndex = [&](int x, int y, int z, int v) {
//                return x + y * dims[0] + z * sliceSize + localVerticesIndex[v];
//            };
//            auto localVertices2GlobalPos = [&](int x, int y, int z, int v) {
//                return std::array<int, 3>{x + localVerticesPos[v][0], y + localVerticesPos[v][1], z + localVerticesPos[v][2]};
//            };
//            for (int z = 0; z < dims[2] - 1; z++) {
//                for (int y = 0; y < dims[1] - 1; y++) {
//                    for (int x = 0; x < dims[0] - 1; x++) {
//                        flag = 0;
//                        int curIndex = x + y * dims[0] + z * sliceSize;
//                        for (int i = 0; i < 8; i++) {
//                            int index = curIndex + localVerticesIndex[i];
//                            if (scalars[index] >= value) {
//                                flag = flag | (1 << i);
//                            }
//                        }
//                        auto edges = cases[flag].edges;
//                        for (int i = 0; i < 16 && edges[i] != -1; i += 3) {
//                            int e[3] = {edges[i], edges[i + 1], edges[i + 2]};
//                            vtkIdType ptIds[3];
//                            for (int j = 0; j < 3; j++) {
//                                int* vertices = edges2Vertices[e[j]];
//                                double F0 = scalars[localVertices2GlobalIndex(x, y, z, vertices[0])];
//                                double F1 = scalars[localVertices2GlobalIndex(x, y, z, vertices[1])];
//                                auto p0 = localVertices2GlobalPos(x, y, z, vertices[0]);
//                                auto p1 = localVertices2GlobalPos(x, y, z, vertices[1]);
//                                double t0 = (value - F0) / (F1 - F0);
//                                double pos[3];
//                                pos[0] = p1[0] * t0 + (1 - t0) * p0[0];
//                                pos[1] = p1[1] * t0 + (1 - t0) * p0[1];
//                                pos[2] = p1[2] * t0 + (1 - t0) * p0[2];
//                                vtkIdType ptId;
//                                if (locator->InsertUniquePoint(pos, ptId)) {
//                                    newScalars->InsertTuple(ptId, &value);
//                                }
//                                ptIds[j] = ptId;
//                            }
//                            if (ptIds[0] != ptIds[1] && ptIds[0] != ptIds[2] && ptIds[1] != ptIds[2])
//                            {
//                                newPolys->InsertNextCell(3, ptIds);
//                            }
//                        }
//                    }
//                }
//            }
//            end = clock();
//            std::cout << "time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
//        }
//    };
//
//
//
//}


int isosurfacesAlgorithm::RequestData(vtkInformation* vtkNotUsed(request),
                                 vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
    // get the info objects
    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
    vtkInformation* outInfo = outputVector->GetInformationObject(0);

    // get the input and output
    vtkImageData* input = vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkPolyData* output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkPoints* newPts;
    vtkCellArray* newPolys;
    vtkFloatArray* newScalars;
    vtkFloatArray* newNormals;
    vtkFloatArray* newGradients;
    vtkPointData* pd;
    vtkDataArray* inScalars;
    int dims[3], extent[6];
    vtkIdType estimatedSize;
    double bounds[6];
    vtkIdType numContours = this->ContourValues->GetNumberOfContours();
    double* values = this->ContourValues->GetValues();

    vtkDebugMacro(<< "Executing marching cubes");

    //
    // Initialize and check input
    //
    pd = input->GetPointData();
    if (pd == nullptr)
    {
        vtkErrorMacro(<< "PointData is nullptr");
        return 1;
    }
    vtkInformationVector* inArrayVec = this->Information->Get(INPUT_ARRAYS_TO_PROCESS());
    if (inArrayVec)
    { // we have been passed an input array
        inScalars = this->GetInputArrayToProcess(0, inputVector);
    }
    else
    {
        inScalars = pd->GetScalars();
    }
    if (inScalars == nullptr)
    {
        vtkErrorMacro(<< "Scalars must be defined for contouring");
        return 1;
    }

    if (inScalars->GetNumberOfComponents() != 1)
    {
        vtkErrorMacro("Scalar array must only have a single component.");
        return 1;
    }

    if (input->GetDataDimension() != 3)
    {
        vtkErrorMacro(<< "Cannot contour data of dimension != 3");
        return 1;
    }
    input->GetDimensions(dims);

    inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent);

    // estimate the number of points from the volume dimensions
    estimatedSize = static_cast<vtkIdType>(pow(1.0 * dims[0] * dims[1] * dims[2], 0.75));
    estimatedSize = estimatedSize / 1024 * 1024; // multiple of 1024
    if (estimatedSize < 1024)
    {
        estimatedSize = 1024;
    }
    vtkDebugMacro(<< "Estimated allocation size is " << estimatedSize);
    newPts = vtkPoints::New();
    newPts->SetDataTypeToDouble();
    newPts->Allocate(estimatedSize, estimatedSize / 2);
    // compute bounds for merging points
    for (int i = 0; i < 3; i++)
    {
        bounds[2 * i] = extent[2 * i];
        bounds[2 * i + 1] = extent[2 * i + 1];
    }
    if (this->Locator == nullptr)
    {
        this->CreateDefaultLocator();
    }
    this->Locator->InitPointInsertion(newPts, bounds, estimatedSize);

    if (this->ComputeNormals)
    {
        newNormals = vtkFloatArray::New();
        newNormals->SetNumberOfComponents(3);
        newNormals->Allocate(3 * estimatedSize, 3 * estimatedSize / 2);
    }
    else
    {
        newNormals = nullptr;
    }

    if (this->ComputeGradients)
    {
        newGradients = vtkFloatArray::New();
        newGradients->SetNumberOfComponents(3);
        newGradients->Allocate(3 * estimatedSize, 3 * estimatedSize / 2);
    }
    else
    {
        newGradients = nullptr;
    }

    newPolys = vtkCellArray::New();
    newPolys->AllocateEstimate(estimatedSize, 3);

    if (this->ComputeScalars)
    {
        newScalars = vtkFloatArray::New();
        newScalars->Allocate(estimatedSize, estimatedSize / 2);
    }
    else
    {
        newScalars = nullptr;
    }

//    using Dispatcher = vtkArrayDispatch::Dispatch;
//    ComputeGradientWorker worker;
//    if (!Dispatcher::Execute(inScalars, worker, this, dims, this->Locator, newScalars, newGradients,
//                             newNormals, newPolys, values, numContours))
//    { // Fallback to slow path for unknown arrays:
//        std::cout << "Fallback to slow path for unknown arrays" << std::endl;
//        worker(inScalars, this, dims, this->Locator, newScalars, newGradients, newNormals, newPolys,
//               values, numContours);
//    }
    std::cout << "ComputeGradientWorker, type:" << inScalars->GetDataType() << std::endl;
    // time begin
    auto start = std::chrono::high_resolution_clock::now();;

    process(inScalars, this, dims, this->Locator, newScalars, newGradients, newNormals, newPolys, values,
                 numContours);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();

    SetLastTimeSpent(seconds);
    std::cout << "函数执行时间: " << seconds << " 秒" << std::endl;

    vtkDebugMacro(<< "Created: " << newPts->GetNumberOfPoints() << " points, "
                          << newPolys->GetNumberOfCells() << " triangles");
#ifdef CAL_TOPOLOGY_ERROR
    calculateTopologyError(newPts, inScalars, dims, values[0]);
#endif
    //
    // Update ourselves.  Because we don't know up front how many triangles
    // we've created, take care to reclaim memory.
    //
    output->SetPoints(newPts);
    newPts->Delete();

    output->SetPolys(newPolys);
    newPolys->Delete();

    vtkNew<vtkRemoveDuplicatePolys> removeDuplicatePolys;
    removeDuplicatePolys->SetInputData(output);
    removeDuplicatePolys->Update();
    output->ShallowCopy(removeDuplicatePolys->GetOutput());

    if (newScalars)
    {
        int idx = output->GetPointData()->AddArray(newScalars);
        output->GetPointData()->SetActiveAttribute(idx, vtkDataSetAttributes::SCALARS);
        newScalars->Delete();
    }
    if (newGradients)
    {
        output->GetPointData()->SetVectors(newGradients);
        newGradients->Delete();
    }
    if (newNormals)
    {
        output->GetPointData()->SetNormals(newNormals);
        newNormals->Delete();
    }
    output->Squeeze();
    if (this->Locator)
    {
        this->Locator->Initialize(); // free storage
    }

//    vtkImageTransform::TransformPointSet(input, output);

    return 1;
}

void isosurfacesAlgorithm::SetLocator(vtkIncrementalPointLocator* locator)
{
    if (this->Locator == locator)
    {
        return;
    }

    if (this->Locator)
    {
        this->Locator->UnRegister(this);
        this->Locator = nullptr;
    }

    if (locator)
    {
        locator->Register(this);
    }

    this->Locator = locator;
    this->Modified();
}

void isosurfacesAlgorithm::CreateDefaultLocator()
{
    if (this->Locator == nullptr)
    {
        this->Locator = vtkMergePoints::New();
    }
}

int isosurfacesAlgorithm::FillInputPortInformation(int, vtkInformation* info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    return 1;
}

void isosurfacesAlgorithm::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);

    this->ContourValues->PrintSelf(os, indent.GetNextIndent());

    os << indent << "Compute Normals: " << (this->ComputeNormals ? "On\n" : "Off\n");
    os << indent << "Compute Gradients: " << (this->ComputeGradients ? "On\n" : "Off\n");
    os << indent << "Compute Scalars: " << (this->ComputeScalars ? "On\n" : "Off\n");

    if (this->Locator)
    {
        os << indent << "Locator:" << this->Locator << "\n";
        this->Locator->PrintSelf(os, indent.GetNextIndent());
    }
    else
    {
        os << indent << "Locator: (none)\n";
    }
}

bool isosurfacesAlgorithm::supportAdaptiveMeshing() {
    return false;
}

//void isosurfacesAlgorithm::process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
//                                   vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
//                                   vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) {
//    using Dispatcher = vtkArrayDispatch::Dispatch;
//    ComputeGradientWorker worker;
//    if (!Dispatcher::Execute(scalarsArray, worker, this, dims, this->Locator, newScalars, newGradients,
//                             newNormals, newPolys, values, numValues))
//    { // Fallback to slow path for unknown arrays:
//        std::cout << "Fallback to slow path for unknown arrays" << std::endl;
//        worker(scalarsArray, this, dims, this->Locator, newScalars, newGradients, newNormals, newPolys,
//               values, numValues);
//    }
//}
//VTK_ABI_NAMESPACE_END
