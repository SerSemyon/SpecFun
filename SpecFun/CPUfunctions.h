#pragma once

long long Fact(int x);

/// <summary>
/// Вычисление гамма-функции при x>-3
/// </summary>
double Gamma(double x);

/// <summary>
/// Вычисление функции порядка v+1 через значения порядка v и v-1
/// </summary>
/// <param name="v"> порядок функции </param>
/// <param name="x"> значения параметра </param>
/// <param name="value_v"> значение порядка v </param>
/// <param name="value_v_minus_1"> значение порядка v-1 </param>
double cyl_next_order(double v, double x, double value_v, double value_v_minus_1);

/// <summary>
/// Вычисление функции Бесселя на CPU для одной точки
/// </summary>
/// <param name="v"> порядок функции </param>
/// <param name="x"> значения параметра </param>
double J(double v, double x);

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
double J_0(double x);

/// <summary>
/// Вычисление функции Бесселя нулевого порядка на отрезке [-8;8] 
/// </summary>
/// <param name="x"> Значение параметра </param>
double J_1(double x);

/// <summary>
/// Вычисление полинома Чебышёва первого рода
/// </summary>
/// <param name="n"> порядок полинома </param>
/// <param name="x"> значения параметра </param>
double T(int n, double x);

/// <summary>
/// Вычисление функции Неймана целого порядка
/// </summary>
/// <param name="v"> Порядок функции </param>
/// <param name="x"> Значение параметра </param>
/// <param name="res"> результат вычислений </param>
/// <param name="n"> количество точек </param>
/// <param name="Jpositive"> значения функции Бесселя порядка v </param>
void Neumann(int v, double* x, double* res, int n, double* Jpositive);

/// <summary>
/// Вычисление функции Неймана
/// </summary>
/// <param name="v"> Порядок функции </param>
/// <param name="x"> Значение параметра </param>
/// <param name="res"> результат вычислений </param>
/// <param name="n"> количество точек </param>
/// <param name="Jpositive"> значения функции Бесселя порядка v </param>
/// <param name="Jnegative"> значения функции Бесселя порядка -v </param>
void Neumann(double v, double* x, double* res, int n, double* Jpositive, double* Jnegative);

/// <summary>
/// Вычисление функции Неймана нулевого порядка на (0;8]
/// </summary>
/// <param name="x"> Значение параметра </param>
/// <param name="J0"> значения функции Бесселя нулевого порядка </param>
double Y_0(double x, double J0);

/// <summary>
/// Вычисление функции Неймана нулевого порядка на (0;8]
/// </summary>
/// <param name="x"> Значение параметра </param>
/// <param name="J1"> значения функции Бесселя первого порядка </param>
double Y_1(double x, double J1);

/// <summary>
/// Мой вариант вычисления функции Бесселя для упорядоченного по возрастанию набора точек
/// </summary>
/// <param name="x"> Значение параметра </param>
/// <param name="v"> Порядок функции </param>
/// <param name="res"> Указатель на результат </param>
/// <param name="n"> Количество точек </param>
void BesselOrderedSet(double* x, double v, double* res, int n);

/// <summary>
/// Мой вариант вычисления функции Бесселя
/// </summary>
/// <param name="x"> Значение параметра </param>
/// <param name="v"> Порядок функции </param>
/// <param name="res"> Указатель на результат </param>
/// <param name="n"> Количество точек </param>
void Jnew(double* x, double v, double* res, int n);