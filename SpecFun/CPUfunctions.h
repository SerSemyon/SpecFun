#pragma once

/// <summary>
/// ���������� �����-������� ��� x>-3
/// </summary>
double Gamma(double x);

/// <summary>
/// ���������� ������� ������� �� CPU
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> ���������� �������� </param>
/// <param name="size"> ���������� ����� </param>
void J(const double v, const double* x, double* result, const unsigned int size);

/// <summary>
/// ���������� �������� �������� ������� ����
/// </summary>
/// <param name="n"> ������� �������� </param>
/// <param name="x"> �������� ��������� </param>
double T(int n, double x);