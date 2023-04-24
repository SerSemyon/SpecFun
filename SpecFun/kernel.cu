#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "CPUfunctions.h"
#include "GPUfunctions.h"
#include "Test.h"
#include "log_duration.h"

int main()
{
    TestBesselCPU();
    TestJ0();
    TestJ1();
    //TestBesselNew();
    TestBessel_CUDA();
    TestJ0_CUDA();
    TestJ1_CUDA();
    TestNeumannCPU();
    TestY0();
    TestY1();
    TestY0_CUDA();
    TestChebyshevPolynomials();
}