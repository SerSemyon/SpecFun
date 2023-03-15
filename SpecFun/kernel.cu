#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

double Gamma(double x) {
    //коэффициенты разложения при |x|<3
    double a[30] = {
        1.0,
        0.57721566490153286061,
        -0.65587807152025388108,
        -0.04200263503409523553,
        0.16653861138229148950,
        -0.04219773455554433675,
        -0.00962197152787697356,
        0.00721894324666309954,
        -0.00116516759185906511,
        -0.00021524167411495097,
        0.00012805028238811619,
        -0.00002013485478078824,
        -0.00000125049348214267,
        0.00000113302723198170,
        -0.00000020563384169776,
        0.00000000611609510448,
        0.00000000500200764447,
        -0.00000000118127457049,
        0.00000000010434267117,
        0.00000000000778226344,
        -0.00000000000369680562,
        0.00000000000051003703,
        -0.00000000000002058326,
        -0.00000000000000534812,
        0.00000000000000122678,
        -0.00000000000000011813,
        0.00000000000000000119,
        0.00000000000000000141,
        -0.00000000000000000023,
        0.00000000000000000002
    };
    //Используем свойство гамма-функции, чтобы перейти в [0,2], на котором разложение даёт верное значение
    double multiplier = 1;
    while (x > 2)
    {
        x--;
        multiplier *= x;
    }
    double sum = 1;
    double z = x - 1;
    double extentZ = 1;
    for (int i = 1; i < 30; i++) {
        extentZ *= z;
        sum += a[i] * extentZ;
    }
    return multiplier / sum;
}

/// <summary>
/// Вычисление функции Бесселя на CPU
/// </summary>
/// <param name="x"> значения параметра </param>
/// <param name="v"> порядок функции </param>
/// <param name="result"> полученные значения </param>
/// <param name="size"> количество точек </param>
void J(const double* x, const double v, double* res, const unsigned int size) {
    double eps = 1E-12;
    double aNext;
    double diff;
    for (int i = 0; i < size; i++) {
        int k = 0;
        double aprev = 1 / Gamma(v + 1);
        double summ = aprev;
        do {
            aNext = -x[i] * x[i] * aprev / ((k + 1) * (v + k + 1) * 4);
            summ += aNext;
            diff = abs(aprev - aNext);
            aprev = aNext;
            k++;
        } while (diff > eps);
        res[i] = summ * pow(x[i] * 0.5, v);
    }
}

/// <summary>
/// Код одной нити GPU
/// </summary>
/// <param name="x"> значения параметра </param>
/// <param name="v"> порядок функции </param>
/// <param name="gamma"> значение гамма функции от (v+1) </param>
/// <param name="result"> полученные значения </param>
__global__ void BesselOneThread(const double* x, const double v, const double gamma, double* result)
{
    int i = blockIdx.x * gridDim.x + threadIdx.x;
    double eps = 1E-12;
    double aNext;
    double diff;
    int k = 0;
    double aprev = 1 / gamma;
    double summ = aprev;
    do {
        aNext = -x[i] * x[i] * aprev / ((k + 1) * (v + k + 1) * 4);
        summ += aNext;
        diff = abs(aprev - aNext);
        aprev = aNext;
        k++;
    } while (diff > eps);
    result[i] = summ * pow(x[i] * 0.5, v);
}

/// <summary>
/// Вычисление функции Бесселя на видеокарте NVidia
/// </summary>
/// <param name="x"> значения параметра </param>
/// <param name="v"> порядок функции </param>
/// <param name="result"> полученные значения </param>
/// <param name="size"> количество точек </param>
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
    BesselOneThread << <1, size >> > (dev_x, v, gamma, dev_res);

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

int main()
{
    int n = 50;
    double* p = new double[n];
    double* y = new double[n];
    for (int i = 0; i < n; i++)
    {
        p[i] = 0.1 * i;
    }
    BesselWithCuda(p, 2, y, n);
    for (int i = 0; i < n; i++)
    {
        std::cout << p[i] << '\t' << y[i] << std::endl;
    }
    return 0;
}