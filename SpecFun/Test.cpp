#pragma once
#include <ctime>
#include "Test.h"
#include "GPUfunctions.h"
#include "CPUfunctions.h"
#include <iostream>

void TestBessel()
{
    int n = 1500000;
    double* x = new double[n];
    double* resGPU = new double[n];
    double* resCPU = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0001 * i;
    }

    unsigned int start_time = clock();
    J(x, 2, resCPU, n);
    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    unsigned int GPUstart_time = clock();
    BesselWithCuda(x, 2, resGPU, n);
    unsigned int GPUend_time = clock();
    unsigned int GPUsearch_time = GPUend_time - GPUstart_time;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "execution time cuda " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "GPU clock:" << GPUsearch_time << " CPU clock:" << search_time << std::endl;

    for (int i = 0; i < n; i++)
    {
        if (abs(resGPU[i] - resCPU[i]) > 1E-12)
        {
            std::cout << "WARNING!!!TestBessel failed! point:" << x[i] << " |resGPU-resCPU|=" << abs(resGPU[i] - resCPU[i])  << std::endl;
            return;
        }
        //std::cout << x[i] << '\t' << resGPU[i] << " " << resCPU[i] << std::endl;
    }
    std::cout << " TestBessel OK" << std::endl;
}