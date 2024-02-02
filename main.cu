#include <iostream>

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "Octree.cuh"
#include <vtkNrrdReader.h>
#include <vtkNew.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <chrono>
#include <vtkOBJWriter.h>
#include "vtkDataArrayRange.h"
#include "common.cuh"
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "gpu_mdmc/myGPUMDMC.cuh"
#include "cuda_tmc/gpuMC.cuh"

#include <filesystem>

void writeToFile(const std::string &i_file, vtkSmartPointer<vtkPolyDataAlgorithm> surface) {
    vtkNew<vtkOBJWriter> writer;
    writer->SetFileName(i_file.c_str());
    writer->SetInputConnection(surface->GetOutputPort());
    writer->Write();
}


int main() {
    ambiguousFacesInit();
    ambiguousCasesComplementInit();
    std::vector<std::pair<std::string, double>> files = {
            {"data/boston_teapot.nhdr", 20.5},
//            {"data/skull.nhdr", 30},
//            {"data/foot.nhdr", 110.3},
//            {"data/vis_male.nhdr", 110.3},
//            {"data/carp.nhdr",1150.5}
    };

    double adaptiveThreshold = 0.f;
    bool forceRewrite = true;
    bool writeConsoleToFile = false;
    for (auto &file : files) {
        auto i_file = file.first;
        double isoValue = file.second;
        std::cout << "Processing" << i_file << std::endl;
        vtkNew<vtkNrrdReader> nrrdReader;
        nrrdReader->SetFileName(i_file.c_str());
        nrrdReader->Update();
        vtkNew<vtkImageData> volume;
        volume->DeepCopy(nrrdReader->GetOutput());
        auto dims = volume->GetDimensions();
//        vtkNew<myGPUMDMC> surface;
        vtkNew<gpuMC> surface;
        surface->SetAdaptiveThreshold(adaptiveThreshold);
        surface->SetInputData(volume);
//    surface->ComputeNormalsOn();
        surface->ComputeNormalsOff();
        surface->SetValue(0, isoValue);


        std::string outputPath = "output";
        std::string consolePath = "console";

        if (!std::filesystem::exists(outputPath)) {
            std::filesystem::create_directories(outputPath);
        }
        if (!std::filesystem::exists(consolePath)) {
            std::filesystem::create_directories(consolePath);
        }
        std::string surfaceType = typeid(*surface).name();
        std::string fileName = std::filesystem::path(i_file).stem();
        std::string outputName = surfaceType + "_" + fileName  + "_"
                                 + std::to_string((int)isoValue) + "_" + std::to_string((int)adaptiveThreshold);
        auto consoleFilename = consolePath + "/" + outputName + ".txt";
        if (!forceRewrite && std::filesystem::exists(consoleFilename)) {
            std::cout << "file already exists, skip:" << consoleFilename << std::endl;
            continue;
        }

        std::ofstream consoleFile(consoleFilename);
        if (!consoleFile || !consoleFile.is_open()) {
            std::cout << "open console file fail" << std::endl;
            continue;
        }
        std::streambuf* originalCoutBuffer = std::cout.rdbuf();
        if (writeConsoleToFile) {
            std::cout.rdbuf(consoleFile.rdbuf()); // 将输出重定向到consoleFile
        }

        {
            surface->Update();
        }

        auto pointCnt = surface->GetOutput()->GetNumberOfPoints();
        auto cellCnt = surface->GetOutput()->GetNumberOfCells();
        std::cout << "Number of surface point: " << pointCnt << std::endl;
        std::cout << "Number of surface cells: " << cellCnt << std::endl;
        if (!pointCnt || !cellCnt) {
            std::cout << "No surface generated" << std::endl;
            continue;
        }

        std::cout.rdbuf(originalCoutBuffer); // 将输出重定向回来
        consoleFile.close();
//        const std::string i_file = "test.obj";
        {
            std::string objName = outputPath + "/" + outputName + ".obj";
            writeToFile(objName, surface);
            std::cout << "Writing " << objName << std::endl;
        }


    }


    cudaFree(ambiguousCasesComplement);
    cudaFree(ambiguousFaces);




    return 0;
}
