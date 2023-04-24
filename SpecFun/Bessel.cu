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
__global__ void BesselOneThread(const double v, const double* x, const double gamma, double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    const double eps = 1E-12;
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

void BesselWithCuda(const double v, const double* const x, double* result, const unsigned int size)
{
    double* dev_x = 0;
    double* dev_res = 0;

    cudaMalloc((void**)&dev_res, size * sizeof(double));
    cudaMalloc((void**)&dev_x, size * sizeof(double));
    cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    double gamma = Gamma(v + 1);
    BesselOneThread << <(size + 127) / 128, 128 >> > (v, dev_x, gamma, dev_res, size);

    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_res, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_res);
    cudaFree(dev_x);
}

///// <summary>
///// ��� ������� ���������� ������� �������
///// </summary>
///// <param name="x"> �������� ��������� </param>
///// <param name="v"> ������� ������� </param>
///// <param name="gamma"> �������� ����� ������� �� (v+1) </param>
///// <param name="result"> ���������� �������� </param>
//__global__ void Jnew(const double v, const double* x, const double gamma, double* result, int N)
//{
//    __shared__ double p[256];
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    double aNext;
//    double diff;
//    double aPrev;
//    double a_0;
//    if (i < 256)
//        p[threadIdx.x] = -1 / (4 * (v + threadIdx.x + 1) * (threadIdx.x + 1));
//    __syncthreads();
//    while (i < N)
//    {
//        a_0 = 1 / gamma;
//        aPrev = a_0;
//        result[i] = a_0;
//        int k = 0;
//        do {
//            aNext = p[k] * aPrev * x[i] * x[i];
//            result[i] += aNext;
//            diff = abs(aPrev - aNext);
//            aPrev = aNext;
//            k++;
//        } while (diff > eps);
//        result[i] *= pow(x[i] * 0.5, v);
//        i += blockDim.x * gridDim.x;
//    }
//}
//cudaError_t BesselWithCudaNew(const double v, const double* x, double* result, const unsigned int size)
//{
//    double* dev_x = 0;
//    double* dev_res = 0;
//    cudaError_t cudaStatus;
//
//    cudaStatus = cudaMalloc((void**)&dev_res, size * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    double gamma = Gamma(v + 1);
//    Jnew << <(size + 255) / 256, 256 >> > (v, dev_x, gamma, dev_res, size);
//
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(result, dev_res, size * sizeof(double), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_res);
//    cudaFree(dev_x);
//
//    return cudaStatus;
//}

/// <summary>
/// ��� ����� ���� GPU
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="v"> ������� ������� </param>
/// <param name="gamma"> �������� ����� ������� �� (v+1) </param>
/// <param name="result"> ���������� �������� </param>
__global__ void J0_OneThread(const double* x, double* result, int size)
{
    const double a0[18] =
    {
        0.15772'79714'74890'11956,
        -0.00872'34423'52852'22129,
        0.26517'86132'03336'80987,
        -0.37009'49938'72649'77903,
        0.15806'71023'32097'26128,
        -0.03489'37694'11408'88516,
        0.00481'91800'69467'60450,
        -0.00046'06261'66206'27505,
        0.00003'24603'28821'00508,
        -0.00000'17619'46907'76215,
        0.00000'00760'81635'92419,
        -0.00000'00026'79253'53056,
        0.00000'00000'78486'96314,
        -0.00000'00000'01943'83469,
        0.00000'00000'00041'25321,
        -0.00000'00000'00000'75885,
        0.00000'00000'00000'01222,
        -0.00000'00000'00000'00017
    };
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    const double eps = 1E-12;
    while (i < size)
    {
        double z = x[i] / 8.0; z = 2.0 * z * z - 1.0;
        double s = 0.0, T0 = 1.0, T1 = z;
        double T;
        s = s + a0[0] * T0 + a0[1] * T1;
        for (int n = 2; n <= 17; n++) {
            T = 2.0 * z * T1 - T0;
            s = s + a0[n] * T;
            T0 = T1; T1 = T;
        };
        result[i] = s;
        i += blockDim.x * gridDim.x;
    }
}

void J0_CUDA(const double* const x, double* result, const unsigned int size)
{
    double* dev_x = 0;
    double* dev_res = 0;

    cudaMalloc((void**)&dev_res, size * sizeof(double));
    cudaMalloc((void**)&dev_x, size * sizeof(double));
    cudaMemcpy(dev_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    J0_OneThread << <(size + 127) / 128, 128 >> > (dev_x, dev_res, size);

    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_res, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_res);
    cudaFree(dev_x);
}