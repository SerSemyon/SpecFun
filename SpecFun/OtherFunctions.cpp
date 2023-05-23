#include <math.h>
#include "CPUfunctions.h"
#include <cmath>

double Z_vNext(double v, double x, double Z_v, double Z_vPrev)
{
	return 2 * v * Z_v / x - Z_vPrev;
}

void dZ(double v, double* x, double* result, unsigned int size, double* Z_vPrev, double* Z_vNext)
{
	for (int i = 0; i < size; i++)
	{
		result[i] = 0.5 * (Z_vPrev[i] - Z_vNext[i]);
	}
}

double dZ(double v, double Z_vPrev, double Z_vNext)
{
	return 0.5 * (Z_vPrev - Z_vNext);
}