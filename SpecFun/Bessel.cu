#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "CPUfunctions.h"
#include "GPUfunctions.h"
#include <iostream>

/// <summary>
/// Код одной нити GPU
/// </summary>
/// <param name="x"> значения параметра </param>
/// <param name="v"> порядок функции </param>
/// <param name="gamma"> значение гамма функции от (v+1) </param>
/// <param name="result"> полученные значения </param>
__global__ void BesselOneThread(const double v, const double* x, const double gamma, double* result, int N)
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
/// Мой вариант вычисления функции Бесселя
/// </summary>
/// <param name="x"> значения параметра </param>
/// <param name="v"> порядок функции </param>
/// <param name="gamma"> значение гамма функции от (v+1) </param>
/// <param name="result"> полученные значения </param>
__global__ void Jnew(const double v, const double* x, const double gamma, double* result, int N)
{
    __shared__ double p[256];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    double eps = 1E-12;
    double aNext;
    double diff;
    double aPrev;
    double a_0;
    if (i < 256)
        p[threadIdx.x] = -1 / (4 * (v + threadIdx.x + 1) * (threadIdx.x + 1));
    __syncthreads();
    while (i < N)
    {
        a_0 = 1 / gamma;
        aPrev = a_0;
        result[i] = a_0;
        int k = 0;
        do {
            aNext = p[k] * aPrev * x[i] * x[i];
            result[i] += aNext;
            diff = abs(aPrev - aNext);
            aPrev = aNext;
            k++;
        } while (diff > eps);
        result[i] *= pow(x[i] * 0.5, v);
        i += blockDim.x * gridDim.x;
    }
}
cudaError_t BesselWithCudaNew(const double v, const double* x, double* result, const unsigned int size)
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
    Jnew << <(size + 255) / 256, 256 >> > (v, dev_x, gamma, dev_res, size);

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

cudaError_t BesselWithCuda(const double v, const double* x, double* result, const unsigned int size)
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
    BesselOneThread << <(size+127)/128, 128 >> > (v, dev_x, gamma, dev_res, size);

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