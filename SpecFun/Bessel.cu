#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "CPUfunctions.h"
#include "GPUfunctions.h"
#include <iostream>

/// <summary>
/// ��� ����� ���� GPU
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="v"> ������� ������� </param>
/// <param name="gamma"> �������� ����� ������� �� (v+1) </param>
/// <param name="result"> ���������� �������� </param>
__global__ void BesselOneThread(const double* x, const double v, const double gamma, double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    double eps = 1E-12;
    double aNext;
    double diff;
    int k;
    double aprev;
    double summ;
    while (i < N)
    {
        k = 0;
        aprev = 1 / gamma;
        summ = aprev;
        do {
            aNext = -x[i] * x[i] * aprev / ((k + 1) * (v + k + 1) * 4);
            summ += aNext;
            diff = abs(aprev - aNext);
            aprev = aNext;
            k++;
        } while (diff > eps);
        result[i] = summ * pow(x[i] * 0.5, v);
        i += blockDim.x * gridDim.x;
    }
}

/// <summary>
/// ���������� ������� ������� �� ���������� NVidia
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="v"> ������� ������� </param>
/// <param name="result"> ���������� �������� </param>
/// <param name="size"> ���������� ����� </param>
cudaError_t BesselWithCuda(const double* x, const double v, double* result, const unsigned int size)
{
    double* dev_x = 0;
    double* dev_res = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_res, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    double gamma = Gamma(v + 1);
    BesselOneThread << <(size+127)/128, 128 >> > (dev_x, v, gamma, dev_res, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(result, dev_res, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_res);
    cudaFree(dev_x);

    return cudaStatus;
}