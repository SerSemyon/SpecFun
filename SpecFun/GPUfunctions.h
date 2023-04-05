#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t BesselWithCuda(const double v, const double* x, double* result, const unsigned int size);