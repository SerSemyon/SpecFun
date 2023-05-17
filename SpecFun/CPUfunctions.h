#pragma once

long long Fact(int x);

/// <summary>
/// ���������� �����-������� ��� x>-3
/// </summary>
double Gamma(double x);

/// <summary>
/// ���������� ������� ������� v+1 ����� �������� ������� v � v-1
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="value_v"> �������� ������� v </param>
/// <param name="value_v_minus_1"> �������� ������� v-1 </param>
double cyl_next_order(double v, double x, double value_v, double value_v_minus_1);

/// <summary>
/// ���������� ������� ������� �� CPU ��� ����� �����
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
double J(double v, double x);

double J_asymptotic(const double v, const double x);

double Y_asymptotic(const double v, const double x);

/// <summary>
/// ���������� ������� ������� �� CPU
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="result"> ���������� �������� </param>
/// <param name="size"> ���������� ����� </param>
void J(const double v, const double* x, double* result, const unsigned int size);

/// <summary>
/// ���������� ������� ������� �������� ������� �� ������� [-8;8] 
/// </summary>
/// <param name="x"> �������� ��������� </param>
double J_0(double x);

/// <summary>
/// ���������� ������� ������� �������� ������� �� ������� [-8;8] 
/// </summary>
/// <param name="x"> �������� ��������� </param>
double J_1(double x);

/// <summary>
/// ���������� �������� �������� ������� ����
/// </summary>
/// <param name="n"> ������� �������� </param>
/// <param name="x"> �������� ��������� </param>
double T(int n, double x);

/// <summary>
/// ���������� ������� ������� ������ �������
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="res"> ��������� ���������� </param>
/// <param name="n"> ���������� ����� </param>
/// <param name="Jpositive"> �������� ������� ������� ������� v </param>
void Neumann(int v, double* x, double* res, int n, double* Jpositive);

/// <summary>
/// ���������� ������� �������
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="res"> ��������� ���������� </param>
/// <param name="n"> ���������� ����� </param>
/// <param name="Jpositive"> �������� ������� ������� ������� v </param>
/// <param name="Jnegative"> �������� ������� ������� ������� -v </param>
void Neumann(double v, double* x, double* res, int n, double* Jpositive, double* Jnegative);

/// <summary>
/// ���������� ������� ������� �������� ������� �� (0;8]
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="J0"> �������� ������� ������� �������� ������� </param>
double Y_0(double x, double J0);

/// <summary>
/// ���������� ������� ������� ������� ������� �� (0;8]
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="J1"> �������� ������� ������� ������� ������� </param>
double Y_1(double x, double J1);

/// <summary>
/// ���������� ������� ������� �������� ������� �� (0;8]
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="res"> ��������� ���������� </param>
/// <param name="n"> ���������� ����� </param>
/// <param name="J0"> �������� ������� ������� �������� ������� </param>
void Y_0(double* x, double* res, int n, double* J0);

/// <summary>
/// ���������� ������� ������� ������� ������� �� (0;8]
/// </summary>
/// <param name="x"> �������� ��������� </param>
/// <param name="res"> ��������� ���������� </param>
/// <param name="n"> ���������� ����� </param>
/// <param name="J1"> �������� ������� ������� ������� ������� </param>
void Y_1(double* x, double* res, int n, double* J1);

/// <summary>
/// ��� ������� ���������� ������� ������� ��� �������������� �� ����������� ������ �����
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="res"> ��������� �� ��������� </param>
/// <param name="n"> ���������� ����� </param>
void BesselOrderedSet(double v, double* x, double* res, int n);

/// <summary>
/// ��� ������� ���������� ������� �������
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="res"> ��������� �� ��������� </param>
/// <param name="n"> ���������� ����� </param>
void Jnew(double v, double* x, double* res, int n);

/// <summary>
/// ��������� �������� ����� �������������� ������� ������� v+1 ��������� �������� ��������� �������� ������� v, v-1.
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="x"> �������� ��������� </param>
/// <param name="Z_v"> �������� ������� ������� v </param>
/// <param name="Z_vPrev"> �������� ������� ������� v-1</param>
/// <returns></returns>
double Z_vNext(double v, double x, double Z_v, double Z_vPrev);

/// <summary>
/// ���������� ����������� ����� �������������� ������� ����� ��������� �������� ������� ������� v-1 � v+1.
/// </summary>
/// <param name="v"> ������� ������� </param>
/// <param name="Z_vPrev"> �������� ������� ������ v-1 </param>
/// <param name="Z_vNext"> �������� ������� ������ v+1 </param>
/// <returns></returns>
double dZ(double v, double Z_vPrev, double Z_vNext);