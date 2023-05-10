#include <math.h>
#include "CPUfunctions.h"
#include <cmath>

double Z_vNext(double v, double x, double Z_v, double Z_vPrev)
{
	return 2 * v * Z_v / x - Z_vPrev;
}

double dZ(double v, double Z_vPrev, double Z_vNext)
{
	return 0.5 * (Z_vPrev - Z_vNext);
}