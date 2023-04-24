#pragma once
#define _USE_MATH_DEFINES
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
__global__ void Y0_OneThread(const double* const x, double* result, int size, const double* const J0)
{
    const double C = 0.5772156;
    const double b0[] = {
        -0.02150'51114'49657'55061,
        -0.27511'81330'43518'79146,
        0.19860'56347'02554'15556,
        0.23425'27461'09021'80210,
        -0.16563'59817'13650'41312,
        0.04462'13795'40669'28217,
        -0.00693'22862'91523'18829,
        0.00071'91174'03752'30309,
        -0.00005'39250'79722'93939,
        0.00000'30764'93288'10848,
        -0.00000'01384'57181'23009,
        0.00000'00050'51054'36909,
        -0.00000'00001'52582'85043,
        0.00000'00000'03882'86747,
        -0.00000'00000'00084'42875,
        0.00000'00000'00001'58748,
        -0.00000'00000'00000'02608,
        0.00000'00000'00000'00038
    };
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < size)
    {
        double T2 = x[i] / 8.0;
        T2 = 2.0 * T2 * T2 - 1.0;
        double T_previous = 1.0; double T_current = T2;
        double T;
        double sum = b0[0] * T_previous + b0[1] * T_current;
        for (int n = 2; n <= 17; n++) {
            T = 2.0 * T2 * T_current - T_previous;
            sum += b0[n] * T;
            T_previous = T_current; T_current = T;
        };
        sum += (log(x[i] / 2.0) + C) * J0[i] * 2.0 / M_PI;
        result[i] = sum;
        i += blockDim.x * gridDim.x;
    }
}

void Y0_CUDA(const double* const x, double* result, const unsigned int size, const double* const J0)
{
    double* dev_x = 0;
    double* dev_res = 0;

    /*cudaMalloc((void**)&dev_res, size * sizeof(double));
    cudaMalloc((void**)&dev_x, size * sizeof(double));
    cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);*/



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

    Y0_OneThread << <(size + 127) / 128, 128 >> > (dev_x, dev_res, size, J0);

    /*cudaError_t cudaStatus = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_res, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_res);
    cudaFree(dev_x);*/

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
}