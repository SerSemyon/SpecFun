#pragma once
#include <ctime>
#include "Test.h"
#include "GPUfunctions.h"
#include "CPUfunctions.h"
#include <iostream>

unsigned int FindExecutionTime(void method())
{
    unsigned int start_time = clock(); // начальное время
    method();
    unsigned int end_time = clock(); // конечное время
    unsigned int search_time = end_time - start_time;
    return search_time;
}

void TestBessel()
{
    int n = 5000;
    double* x = new double[n];
    double* resGPU = new double[n];
    double* resCPU = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.01 * i;
    }
    BesselWithCuda(x, 2, resGPU, n);
    J(x, 2, resCPU, n);
    for (int i = 0; i < n; i++)
    {
        if (abs(resGPU[i] - resCPU[i]) > 1E-12)
        {
            std::cout << "WARNING!!!TestBessel failed! " << i << " " << abs(resGPU[i] - resCPU[i])  << std::endl;
            return;
        }
        //std::cout << x[i] << '\t' << resGPU[i] << " " << resCPU[i] << std::endl;
    }
    std::cout << " TestBessel OK" << std::endl;
}