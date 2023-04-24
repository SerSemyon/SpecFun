#pragma once
#include <ctime>
#include "Test.h"
#include "GPUfunctions.h"
#include "CPUfunctions.h"
#include <iostream>
#include "log_duration.h"

double epsilon = 1E-12;

/* TODO - Встроенная реализация даёт низкую точность и, чем дальше от нуля, тем выше ошибка.
Поэтому для проверки значений нужно будет использовать таблицы с более точными результатами,иначе тест всегда будет проваливаться. 
Текущая реализация теста говорит, что тест провален, даже если значения верны. */
void TestBesselCPU()
{
    std::cout << "TestBesselCPU started" << std::endl;
    int v = 1;
    int n = 1000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.01;
        res1[i] = std::cyl_bessel_i(v, x[i]);//__std_smf_cyl_bessel_i(v, x[i]);
    }
    J(v, x, res2, n);
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > epsilon)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestBesselCPU failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestBesselCPU OK" << std::endl << std::endl;
}

void TestJ0()
{
    std::cout << "TestJ0 started" << std::endl;
    int v = 0;
    int n = 1000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.001;
    }
    {
        LOG_DURATION("J");
        J(v, x, res1, n);
    }
    {
        LOG_DURATION("J0");
        for (int i = 0; i < n; i++)
        {
            res2[i] = J_0(x[i]);
        }
    }
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > epsilon)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestJ0 failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestJ0 OK" << std::endl << std::endl;
}

void TestJ1()
{
    std::cout << "TestJ1 started" << std::endl;
    int v = 1;
    int n = 1000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.001;
    }
    {
        LOG_DURATION("J");
        J(v, x, res1, n);
    }
    {
        LOG_DURATION("J1");
        for (int i = 0; i < n; i++)
        {
            res2[i] = J_1(x[i]);
        }
    }
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > epsilon)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestJ1 failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestJ1 OK" << std::endl << std::endl;
}

void TestNeumannCPU()
{
    std::cout << "TestNeumannCPU started" << std::endl;
    int v = 0;
    int n = 1000000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.00000001;
        res1[i] = __std_smf_cyl_neumann(v, x[i]);
    }
    double* Js = new double[n];
    J(v, x, Js, n);
    {
        LOG_DURATION("Neumann");
        Neumann(v, x, res2, n, Js);
    }
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > 1E-4)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestNeumannCPU failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestNeumannCPU OK" << std::endl << std::endl;
}

void TestY0()
{
    std::cout << "TestY0 started" << std::endl;
    int v = 0;
    int n = 1000000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.00000001;
    }
    double* Js = new double[n];
    J(v, x, Js, n);
    {
        LOG_DURATION("Y_0");
        for (int i = 0; i < n; i++)
        {
            res1[i] = Y_0(x[i], Js[i]);
        }
    }
    {
        LOG_DURATION("Neumann");
        Neumann(v, x, res2, n, Js);
    }
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > 1E-4)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestY0 failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestY0 OK" << std::endl << std::endl;
}

void TestY1()
{
    std::cout << "TestY1 started" << std::endl;
    int v = 1;
    int n = 1000000;
    bool successfully = true;
    double* res1 = new double[n];
    double* res2 = new double[n];
    double* x = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = i * 0.00000001;
    }
    double* Js = new double[n];
    J(v, x, Js, n);
    {
        LOG_DURATION("Y_1");
        for (int i = 0; i < n; i++)
        {
            res1[i] = Y_1(x[i], Js[i]);
        }
    }
    {
        LOG_DURATION("Neumann");
        Neumann(v, x, res2, n, Js);
    }
    for (int i = 0; i < n; i++)
    {
        if (abs(res1[i] - res2[i]) > 1E-4)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestY1 failed!" << x[i] << " " << res1[i] << " " << res2[i] << std::endl << std::endl;
            successfully = false;
            break;
        }
    }
    if (successfully)
        std::cout << "TestY1 OK" << std::endl << std::endl;
}

void TestBesselCuda()
{
    std::cout << "TestBesselCuda started" << std::endl;
    int n = 1500000;
    double* x = new double[n];
    double* resGPU = new double[n];
    double* resCPU = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0001 * i;
    }

    {
        LOG_DURATION("CPU clock");
        J(2, x, resCPU, n);
    }

    {
        LOG_DURATION("GPU clock");
        BesselWithCuda(2, x, resGPU, n);
    }

    for (int i = 0; i < n; i++)
    {
        if (abs(resGPU[i] - resCPU[i]) > epsilon)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "WARNING!!!TestBesselCuda failed! point:" << x[i] << " |resGPU-resCPU|=" << abs(resGPU[i] - resCPU[i])  << std::endl << std::endl;
            return;
        }
        //std::cout << x[i] << '\t' << resGPU[i] << " " << resCPU[i] << std::endl;
    }
    std::cout << "TestBesselCuda OK" << std::endl << std::endl;
}

void TestBesselNew() {
    std::cout << "TestBesselNew started" << std::endl;
    int n = 256;
    double* x = new double[n];
    double* resGPU = new double[n];
    double* resCPU = new double[n];
    for (int i = 0; i < n; i++)
    {
        x[i] = 0.0001 * i;
    }

    {
        LOG_DURATION("CPU clock");
        J(2, x, resCPU, n);
    }

    {
        LOG_DURATION("GPU clock");
        BesselWithCudaNew(2, x, resGPU, n);
    }

    for (int i = 0; i < n; i++)
    {
        if (abs(resGPU[i] - resCPU[i]) > 1E-13)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestBessel failed! point:" << x[i] << " |resGPU-resCPU|=" << abs(resGPU[i] - resCPU[i]) << std::endl << std::endl;
            return;
        }
        //std::cout << x[i] << '\t' << resGPU[i] << " " << resCPU[i] << std::endl;
    }
    std::cout << "TestBesselNew OK" << std::endl << std::endl;
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
    LOG_DURATION("TestChebyshevPolynomials");
    std::cout << "TestChebyshevPolynomials started" << std::endl;
    double t1, t2;
    bool successfully = true;
    for (int i = 0; i < 11; i++)
    {
        t1 = T_recursively(i, 0.2);
        t2 = T(i, 0.2);
        if ((t1 - t2) > epsilon)
        {
            std::cout << "WARNING!!!" << std::endl;
            std::cout << "TestChebyshevPolynomials failed! order " << i << "T_recursively - T = " << t1 - t2 << std::endl << std::endl;
            successfully = false;
        }
    }
    if (successfully)
        std::cout << "TestChebyshevPolynomials OK" << std::endl << std::endl;
}