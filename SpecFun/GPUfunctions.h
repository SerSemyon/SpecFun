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
void BesselWithCuda(const double v, const double* const x, double* result, const unsigned int size);
//cudaError_t BesselWithCudaNew(const double v, const double* x, double* result, const unsigned int size);

/// <summary>
/// ���������� ������� ������� �������� ������� �� ������� [-8;8] 
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> ���������� �������� </param>
/// <param name="size"> ���������� ����� </param>
void J0_CUDA(const double* const x, double* result, const unsigned int size);

/// <summary>
/// ���������� ������� ������� �������� ������� �� ������� [-8;8] 
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> ���������� �������� </param>
/// <param name="size"> ���������� ����� </param>
void J1_CUDA(const double* const x, double* result, const unsigned int size);

/// <summary>
/// ���������� ������� ������� �������� ������� �� (0;8]
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> �������� ��������� </param>
/// <param name="size"> ���������� ����� </param>
/// <param name="J0"> �������� ������� ������� �������� ������� </param>
void Y0_CUDA(const double* const x, double* result, const unsigned int size, const double* const J0);

/// <summary>
/// ���������� ������� ������� �������� ������� �� (0;8]
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> �������� ��������� </param>
/// <param name="size"> ���������� ����� </param>
/// <param name="J1"> �������� ������� ������� ������� ������� </param>
void Y1_CUDA(const double* const x, double* result, const unsigned int size, const double* const J1);

void dZ_CUDA(double* x, double* result, unsigned int size, double* Z_vPrev, double* Z_vNext);