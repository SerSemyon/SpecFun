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
    //TestBessel_CUDA();
    //TestJnew();
    //TestBesselOrderedSet();
    //TestJ0();
    //TestJ0_CUDA();
    //TestJ1();
    //TestJ1_CUDA();
    ////TestBesselNew();
    //TestNeumannCPU();
    //TestY0();
    //TestY1();
    //TestY0_CUDA();
    //TestY1_CUDA();
    TestZ_vNext();
    //TestChebyshevPolynomials();
    TestJ_asymptotic();
    TestY_asymptotic();
}