#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// Вычисление функции Бесселя на видеокарте NVidia
/// </summary>
/// <param name="v"> порядок функции </param>
/// <param name="x"> значения параметра </param>
/// <param name="result"> полученные значения </param>
/// <param name="size"> количество точек </param>
cudaError_t BesselWithCuda(const double v, const double* x, double* result, const unsigned int size);