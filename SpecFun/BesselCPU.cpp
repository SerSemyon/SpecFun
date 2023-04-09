#include <math.h>
#include "CPUfunctions.h"
#include <cmath>

void J(const double v, const double* x, double* result, const unsigned int size) {
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
        result[i] = summ * pow(x[i] * 0.5, v);
    }
}

void J_0(const double* const x, double* res, const unsigned int n)
{
	double a[18] =
	{
		0.15772'79714'74890'11956,
		-0.00872'34423'52852'22129,
		0.26517'86132'03336'80987,
		-0.37009'49938'72649'77903,
		0.15806'71023'32097'26128,
		-0.03489'37694'11408'88516,
		0.00481'91800'69467'60450,
		-0.00046'06261'66206'27505,
		0.00003'24603'28821'00508,
		-0.00000'17619'46907'76215,
		0.00000'00760'81635'92419,
		-0.00000'00026'79253'53056,
		0.00000'00000'78486'96314,
		-0.00000'00000'01943'83469,
		0.00000'00000'00041'25321,
		-0.00000'00000'00000'75885,
		0.00000'00000'00000'01222,
		-0.00000'00000'00000'00017
	};
	for (int i = 0; i < n; i++)
	{
		res[i] = a[0];
	}
	for (int k = 1; k < 18; k++)
	{
		for (int i = 0; i < n; i++)
		{
			res[i] += a[k] * T(2 * k, 0.125 * x[i]);
		}
	}
}

void J_1(const double* const x, double* res, const unsigned int n)
{
	double a[18] =
	{
		0.05245'81903'34656'48458,
		0.04809'64691'58230'37394,
		0.31327'50823'61567'18380,
		-0.24186'74084'47407'48475,
		0.07426'67962'16787'03781,
		-0.01296'76273'11735'17510,
		0.00148'99128'96667'63839,
		-0.00012'22786'85054'32427,
		0.00000'75626'30229'69605,
		-0.00000'03661'30855'23363,
		0.00000'00142'77324'38731,
		-0.00000'00004'58570'03076,
		0.00000'00000'12351'74811,
		-0.00000'00000'00283'17735,
		0.00000'00000'00005'59509,
		-0.00000'00000'00000'09629,
		0.00000'00000'00000'00146,
		-0.00000'00000'00000'00002
	};
	for (int i = 0; i < n; i++)
	{
		res[i] = 0.125 * x[i] * a[0];
	}
	for (int k = 1; k < 18; k++)
	{
		for (int i = 0; i < n; i++)
		{
			res[i] += a[k] * T(2 * k + 1, 0.125 * x[i]);
		}
	}
}
