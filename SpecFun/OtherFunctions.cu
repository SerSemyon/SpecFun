#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "CPUfunctions.h"
#include "GPUfunctions.h"
#include <iostream>
#include "log_duration.h"

__global__ void dZ_OneThread(double* x, double* result, unsigned int size, double* Z_vPrev, double* Z_vNext)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < size)
    {
        result[i] = 0.5 * (Z_vPrev[i] - Z_vNext[i]);
        i += blockDim.x * gridDim.x;
    }
}

void dZ_CUDA(double* x, double* result, unsigned int size, double* Z_vPrev, double* Z_vNext)
{
    double* dev_x = 0;
    double* dev_res = 0;
    double* dev_Z_vPrev = 0;
    double* dev_Z_vNext = 0;

    cudaMalloc((void**)&dev_x, size * sizeof(double));
    cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_res, size * sizeof(double));

    cudaMalloc((void**)&dev_Z_vPrev, size * sizeof(double));
    cudaMemcpy(dev_Z_vPrev, Z_vPrev, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_Z_vNext, size * sizeof(double));
    cudaMemcpy(dev_Z_vNext, Z_vNext, size * sizeof(double), cudaMemcpyHostToDevice);

    {
        LOG_DURATION("GPU without data transfers");
        dZ_OneThread << <(size + 127) / 128, 128 >> > (dev_x, dev_res, size, dev_Z_vPrev, dev_Z_vNext);

        cudaGetLastError();
        cudaDeviceSynchronize();
    }

    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(result, dev_res, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_res);
    cudaFree(dev_x);
    cudaFree(dev_Z_vPrev);
    cudaFree(dev_Z_vNext);
}