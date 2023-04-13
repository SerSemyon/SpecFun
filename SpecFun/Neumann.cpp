#define _USE_MATH_DEFINES
#include <math.h>
#include "CPUfunctions.h"
#include <cmath>

void Neumann(int v, double* x, double* res, int n, double* Jpositive)
{
	double C = 0.5772156;
	unsigned int factKN = Fact(v);
	double S_2 = 0;
	unsigned int factK = 1;
	int sign = 1;
	double M = 1.0 / Fact(v);
	for (int i = 1; i <= v; i++)
	{
		S_2 += 1.0 / i;
	}
	M *= S_2;
	double* sumWithPsi = new double[n];
	for (int i = 0; i < n; i++)
	{
		sumWithPsi[i] = M;
	}
	for (int k = 1; k < 15; k++)
	{
		sign = -sign;
		factK *= k;
		factKN *= (v + k);
		S_2 += 1.0 / k + 1.0 / (k + v);
		M = sign * S_2 / (factK * factKN);
		for (int i = 0; i < n; i++)
		{
			sumWithPsi[i] += std::pow(0.5 * x[i], 2 * k) * M;
		}
	}
	for (int i = 0; i < n; i++)
	{
		sumWithPsi[i] *= std::pow(0.5 * x[i], v);
	}
	double* S_1 = new double[n];
	double f_1 = Fact(v - 1);
	for (int i = 0; i < n; i++)
	{
		S_1[i] = f_1;
	}
	for (int k = 1; k < v; k++)
	{
		f_1 *= (v - k - 1);
		f_1 /= k;
		for (int i = 0; i < n; i++)
		{
			S_1[i] += f_1 * std::pow(0.5 * x[i], 2 * k);
		}
	}
	for (int i = 0; i < n; i++)
	{
		res[i] = 2 * Jpositive[i] * (std::log(0.5 * x[i]) + C);
		res[i] -= sumWithPsi[i];
		if (v > 0)
			res[i] -= std::pow(0.5 * x[i], -v) * S_1[i];
		res[i] /= M_PI;
	}
}

void Neumann(double v, double* x, double* res, int n, double* Jpositive, double* Jnegative)
{
	double arg = v * M_PI;
	if (v != (long long)v) // Если не целое
	{
		for (int i = 0; i < n; i++)
		{
			res[i] = (Jpositive[i] * cos(arg) - Jnegative[i]) / sin(arg);
		}
	}
	else
	{
		Neumann(v, x, res, n, Jpositive);
	}
}

double Y_0(double x) {
	double C = 0.5772156;
	double b[] = {
		-0.02150'51114'49657'55061,
		-0.27511'81330'43518'79146,
		0.19860'56347'02554'15556,
		0.23425'27461'09021'80210,
		-0.16563'59817'13650'41312,
		0.04462'13795'40669'28217,
		-0.00693'22862'91523'18829,
		0.00071'91174'03752'30309,
		-0.00005'39250'79722'93939,
		0.00000'30764'93288'10848,
		-0.00000'01384'57181'23009,
		0.00000'00050'51054'36909,
		-0.00000'00001'52582'85043,
		0.00000'00000'03882'86747,
		-0.00000'00000'00084'42875,
		0.00000'00000'00001'58748,
		-0.00000'00000'00000'02608,
		0.00000'00000'00000'00038
	};
	double x0 = x;
	x = x / 8.0; x = 2.0 * x * x - 1.0;
	double s = 0.0;
	double T0 = 1.0; double T1 = x;
	double T;
	s = s + b[0] * T0 + b[1] * T1;
	for (int n = 2; n <= 17; n++) {
		T = 2.0 * x * T1 - T0;
		s = s + b[n] * T;
		T0 = T1; T1 = T;
	};
	s = s + (log(x0 / 2.0) + C) * J_0(x0) * 2.0 / M_PI;
	return s;
};

void Y_0(const double* const x, double* res, int n, const double* const J0)
{
	double C = 0.5772156;
	double b[] = {
		-0.02150'51114'49657'55061,
		-0.27511'81330'43518'79146,
		0.19860'56347'02554'15556,
		0.23425'27461'09021'80210,
		-0.16563'59817'13650'41312,
		0.04462'13795'40669'28217,
		-0.00693'22862'91523'18829,
		0.00071'91174'03752'30309,
		-0.00005'39250'79722'93939,
		0.00000'30764'93288'10848,
		-0.00000'01384'57181'23009,
		0.00000'00050'51054'36909,
		-0.00000'00001'52582'85043,
		0.00000'00000'03882'86747,
		-0.00000'00000'00084'42875,
		0.00000'00000'00001'58748,
		-0.00000'00000'00000'02608,
		0.00000'00000'00000'00038
	};
	for (int i = 0; i < n; i++)
	{
		res[i] = 2.0 / M_PI * (C + log(0.5 * x[i])) * J_0(x[i]);//J0[i];
	}
	for (int k = 0; k < 18; k++)
	{
		for (int i = 0; i < n; i++)
		{
			res[i] += b[k] * T(2 * k, 0.125 * x[i]);
		}
	}
}

void Y_1(const double* const x, double* res, int n, const double* const J1)
{
	double C = 0.5772156;
	double b[] = {
		-0.04017'29465'44414'07579,
		-0.44444'71476'30558'06261,
		-0.02271'92444'28417'73587,
		0.20664'45410'17490'51976,
		-0.08667'16970'56948'52366,
		0.01763'67030'03163'13441,
		-0.00223'56192'94485'09524,
		0.00019'70623'02701'54078,
		-0.00001'28858'53299'24086,
		0.00000'06528'47952'35852,
		-0.00000'00264'50737'17479,
		0.00000'00008'78030'11712,
		-0.00000'00000'24343'27870,
		0.00000'00000'00572'61216,
		-0.00000'00000'00011'57794,
		0.00000'00000'00000'20347,
		-0.00000'00000'00000'00314,
		0.00000'00000'00000'00004
	};
	for (int i = 0; i < n; i++)
	{
		res[i] = 2.0 / M_PI * (C + log(0.5 * x[i])) * J1[i] - 2.0 / (M_PI * x[i]);
	}
	for (int k = 0; k < 18; k++)
	{
		for (int i = 0; i < n; i++)
		{
			res[i] += b[k] * T(2 * k + 1, 0.125 * x[i]);
		}
	}
}