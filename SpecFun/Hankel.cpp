#include "CPUfunctions.h"

void H1(const int v, double* x, double* Re, double* Im, const unsigned int size)
{
	J(v, x, Re, size);
	Neumann(v, x, Im, size, Re);
}