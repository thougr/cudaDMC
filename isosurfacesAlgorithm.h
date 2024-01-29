//
// Created by hougr t on 2023/10/27.
//

#ifndef MC_ISOSURFACESALGORITHM_H
#define MC_ISOSURFACESALGORITHM_H

#include "vtkFiltersCoreModule.h" // For export macro
#include "vtkPolyDataAlgorithm.h"

#include "vtkContourValues.h" // Needed for direct access to ContourValues

class vtkIncrementalPointLocator;

class VTKFILTERSCORE_EXPORT isosurfacesAlgorithm: public vtkPolyDataAlgorithm {

public:
//    static isosurfacesAlgorithm* New();
vtkTypeMacro(isosurfacesAlgorithm, vtkPolyDataAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    // Methods to set contour values
    void SetValue(int i, double value);
    double GetValue(int i);
    double* GetValues();
    void GetValues(double* contourValues);
    void SetNumberOfContours(int number);
    vtkIdType GetNumberOfContours();
    void GenerateValues(int numContours, double range[2]);
    void GenerateValues(int numContours, double rangeStart, double rangeEnd);

    // Because we delegate to vtkContourValues
    vtkMTimeType GetMTime() override;
    vtkSetMacro(ComputeNormals, vtkTypeBool);
    vtkGetMacro(ComputeNormals, vtkTypeBool);
    vtkBooleanMacro(ComputeNormals, vtkTypeBool);

    vtkSetMacro(ComputeGradients, vtkTypeBool);
    vtkGetMacro(ComputeGradients, vtkTypeBool);
    vtkBooleanMacro(ComputeGradients, vtkTypeBool);

    vtkSetMacro(ComputeScalars, vtkTypeBool);
    vtkGetMacro(ComputeScalars, vtkTypeBool);

    vtkSetMacro(LastTimeSpent, vtkTypeFloat64);
    vtkGetMacro(LastTimeSpent, vtkTypeFloat64);
    vtkSetMacro(AdaptiveThreshold, vtkTypeFloat64);
    vtkGetMacro(AdaptiveThreshold, vtkTypeFloat64);
    vtkBooleanMacro(ComputeScalars, vtkTypeBool);

    void SetLocator(vtkIncrementalPointLocator* locator);
    vtkGetObjectMacro(Locator, vtkIncrementalPointLocator);

    void CreateDefaultLocator();

    virtual void process(vtkDataArray* scalarsArray, isosurfacesAlgorithm* self, int dims[3],
                    vtkIncrementalPointLocator* locator, vtkDataArray* newScalars, vtkDataArray* newGradients,
                    vtkDataArray* newNormals, vtkCellArray* newPolys, double* values, vtkIdType numValues) = 0;

    virtual bool supportAdaptiveMeshing();

protected:
    isosurfacesAlgorithm();
    ~isosurfacesAlgorithm() override;

    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    int FillInputPortInformation(int port, vtkInformation* info) override;

    vtkContourValues* ContourValues;
    vtkTypeBool ComputeNormals;
    vtkTypeBool ComputeGradients;
    vtkTypeBool ComputeScalars;
    vtkTypeFloat64 LastTimeSpent;
    vtkIncrementalPointLocator* Locator;
    vtkTypeFloat64 AdaptiveThreshold;

private:
    isosurfacesAlgorithm(const isosurfacesAlgorithm&) = delete;
    void operator=(const isosurfacesAlgorithm&) = delete;
};

/**
 * Set a particular contour value at contour number i. The index i ranges
 * between 0<=i<NumberOfContours.
 */
inline void isosurfacesAlgorithm::SetValue(int i, double value)
{
    this->ContourValues->SetValue(i, value);
}

/**
 * Get the ith contour value.
 */
inline double isosurfacesAlgorithm::GetValue(int i)
{
    return this->ContourValues->GetValue(i);
}

/**
 * Get a pointer to an array of contour values. There will be
 * GetNumberOfContours() values in the list.
 */
inline double* isosurfacesAlgorithm::GetValues()
{
    return this->ContourValues->GetValues();
}

/**
 * Fill a supplied list with contour values. There will be
 * GetNumberOfContours() values in the list. Make sure you allocate
 * enough memory to hold the list.
 */
inline void isosurfacesAlgorithm::GetValues(double* contourValues)
{
    this->ContourValues->GetValues(contourValues);
}

/**
 * Set the number of contours to place into the list. You only really
 * need to use this method to reduce list size. The method SetValue()
 * will automatically increase list size as needed.
 */
inline void isosurfacesAlgorithm::SetNumberOfContours(int number)
{
    this->ContourValues->SetNumberOfContours(number);
}

/**
 * Get the number of contours in the list of contour values.
 */
inline vtkIdType isosurfacesAlgorithm::GetNumberOfContours()
{
    return this->ContourValues->GetNumberOfContours();
}

/**
 * Generate numContours equally spaced contour values between specified
 * range. Contour values will include min/max range values.
 */
inline void isosurfacesAlgorithm::GenerateValues(int numContours, double range[2])
{
    this->ContourValues->GenerateValues(numContours, range);
}

/**
 * Generate numContours equally spaced contour values between specified
 * range. Contour values will include min/max range values.
 */
inline void isosurfacesAlgorithm::GenerateValues(int numContours, double rangeStart, double rangeEnd)
{
    this->ContourValues->GenerateValues(numContours, rangeStart, rangeEnd);
}

#endif //MC_ISOSURFACESALGORITHM_H
