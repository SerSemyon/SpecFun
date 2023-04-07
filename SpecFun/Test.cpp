#pragma once
#include <ctime>
#include "Test.h"
#include "GPUfunctions.h"
#include "CPUfunctions.h"
#include <iostream>
#include "log_duration.h"

double epsilon = 1E-13;

/* TODO - Встроенная реализация даёт низкую точность и, чем дальше от нуля, тем выше ошибка.
Поэтому для проверки значений нужно будет использовать таблицы с более точными результатами,иначе тест всегда будет проваливаться. 
Текущая реализация теста говорит, что тест провален, даже если значения верны. */
void TestBesselCPU()
{
    int v = 1;
    int n = 1000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.01;
        res1[i] = __std_smf_cyl_bessel_i(v, x[i]);
    }
    J(v, x, res2, n);
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > epsilon)
        {
            std::cout << "TestBesselCPU failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestBesselCPU OK" << std::endl;
}

void TestBesselCuda()
{
    LOG_DURATION_H("CPU clock");
    int n = 1500000;
    double* x = new double[n];
    double* resGPU = new double[n];
    double* resCPU = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0001 * i;
    }

    {
        LOG_DURATION_H("CPU clock");
        J(2, x, resCPU, n);
    }

    {
        LOG_DURATION_H("GPU clock");
        BesselWithCuda(2, x, resGPU, n);
    }

    for (int i = 0; i < n; i++)
    {
        if (abs(resGPU[i] - resCPU[i]) > epsilon)
        {
            std::cout << "WARNING!!!TestBessel failed! point:" << x[i] << " |resGPU-resCPU|=" << abs(resGPU[i] - resCPU[i])  << std::endl;
            return;
        }
        //std::cout << x[i] << '\t' << resGPU[i] << " " << resCPU[i] << std::endl;
    }
    std::cout << "TestBessel OK" << std::endl;
}

double T_recursively(int n, double x)
{
    if (n == 0)
        return 1;
    if (n == 1)
        return x;
    return 2 * x * T_recursively(n - 1, x) - T_recursively(n - 2, x);
}

void TestChebyshevPolynomials()
{
    double t1, t2;
    bool successfully = true;
    for (int i = 0; i < 11; i++)
    {
        t1 = T_recursively(i, 0.2);
        t2 = T(i, 0.2);
        if ((t1 - t2) > epsilon)
        {
            std::cout << "WARNING!!!TestChebyshevPolynomials failed! order " << i << "T_recursively - T = " << t1 - t2 << std::endl;
            successfully = false;
        }
    }
    if (successfully)
        std::cout << "TestChebyshevPolynomials OK" << std::endl;
}