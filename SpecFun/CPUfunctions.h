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
/// Вычисление полинома Чебышёва первого рода
/// </summary>
/// <param name="n"> порядок полинома </param>
/// <param name="x"> значения параметра </param>
double T(int n, double x);