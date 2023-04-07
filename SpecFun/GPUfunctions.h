#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// ���������� ������� ������� �� ���������� NVidia
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> ���������� �������� </param>
/// <param name="size"> ���������� ����� </param>
cudaError_t BesselWithCuda(const double v, const double* x, double* result, const unsigned int size);