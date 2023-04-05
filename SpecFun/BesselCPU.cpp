#include <math.h>
#include "CPUfunctions.h"
#include <cmath>

/// <summary>
/// Вычисление функции Бесселя на CPU
/// </summary>
/// <param name="x"> значения параметра </param>
/// <param name="v"> порядок функции </param>
/// <param name="result"> полученные значения </param>
/// <param name="size"> количество точек </param>
void J(const double v, const double* x, double* res, const unsigned int size) {
    double eps = 1E-12;
    double aNext;
    double diff;
    for (int i = 0; i < size; i++) {
        int k = 0;
        double aprev = 1 / Gamma(v + 1);
        double summ = aprev;
        do {
            aNext = -x[i] * x[i] * aprev / ((k + 1) * (v + k + 1) * 4);
            summ += aNext;
            diff = abs(aprev - aNext);
            aprev = aNext;
            k++;
        } while (diff > eps);
        res[i] = summ * pow(x[i] * 0.5, v);
    }
}