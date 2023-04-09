#pragma once

/// <summary>
/// Вычисление гамма-функции при x>-3
/// </summary>
double Gamma(double x);

/// <summary>
/// Вычисление функции Бесселя на CPU
/// </summary>
/// <param name="v"> порядок функции </param>
/// <param name="x"> значения параметра </param>
/// <param name="result"> полученные значения </param>
/// <param name="size"> количество точек </param>
void J(const double v, const double* x, double* result, const unsigned int size);

/// <summary>
/// Вычисление функции Бесселя нулевого порядка на отрезке [-8;8] 
/// </summary>
/// <param name="x"> Значение параметра </param>
/// <param name="res"> Указатель на результат </param>
/// <param name="n"> Количество точек </param>
void J_0(const double* const x, double* res, const unsigned int n);

/// <summary>
/// Вычисление функции Бесселя первого порядка на отрезке [-8;8] 
/// </summary>
/// <param name="x"> Значение параметра </param>
/// <param name="res"> Указатель на результат </param>
/// <param name="n"> Количество точек </param>
void J_1(const double* const x, double* res, const unsigned int n);

/// <summary>
/// Вычисление полинома Чебышёва первого рода
/// </summary>
/// <param name="n"> порядок полинома </param>
/// <param name="x"> значения параметра </param>
double T(int n, double x);