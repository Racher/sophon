#pragma once
#pragma warning( disable : 6294 )
#pragma warning( disable : 26444 )
#include "point.h"

constexpr auto xMidConv = 1.0;
constexpr auto xIntConv = 1.0;
constexpr auto midConv = 0.5;
constexpr auto dConv = 0.06;
constexpr auto d1Min = 0.0;
constexpr auto I1 = size_t(1);
constexpr auto min(double a, double b) { return a < b ? a : b; }
constexpr auto max(double a, double b) { return a > b ? a : b; }
constexpr auto sgn(double a) { return (0 < a) - (a < 0); }

template<size_t size, size_t pieces, size_t order>
void Predict(Point * data) {
	constexpr size_t segments = (I1 << order) - 1;
	for (size_t r = 0; r < order - 1; ++r)
		for (size_t m = (I1 << r) - 1; m < segments / 2; m += I1 << (r + 1)) {
			size_t sLeft = (2 * m + 1) * pieces;
			double lam = -data[(sLeft + 1) * size].a * pieces;
			for (size_t k = 0; k <= r; ++k) {
				double klam = k != 0 ? min(lam - k + 1, 1.0) : 0;
				T avg = T0;
				if (klam > 0) {
					size_t dist = pieces * (I1 << k);
					avg = (data[(sLeft - dist) * size + k].x + data[(sLeft + dist) * size + k].x) * (klam / 2);
				}
				else
					klam = 0;
				for (size_t s = 0; s <= pieces; ++s) {
					size_t offset = (sLeft + s) * size;
					T targ = avg + data[offset + k + 1].z * (-1.0 / data[offset + k + 1].a * (1 - klam));
					data[offset + k].x *= (1 - xMidConv);
					data[offset + k].x += (targ * xMidConv);
				}
			}
		}
	for (size_t r = 0; r < order; ++r) {
		size_t sCount = ((I1 << (r + 1)) - 1) * pieces;
		size_t n0 = (I1 << r) - 1;
		for (size_t n = n0; n < segments; n += I1 << (r + 1)) {
			size_t sLeft = (n - n0) * pieces;
			T zInt = data[sLeft * size + r + 1].x;
			double aInt = 0.0;
			double d = 0.0;
			for (size_t si = 0; ; ++si) {
				size_t s = sLeft + si;
				size_t offset = s * size + r + 1;
				T z = data[offset].z + data[s * size + r].x * data[offset].a;
				zInt += z * d;
				aInt += data[offset].a * d;
				if (si == sCount)
					break;
				data[offset].x *= (1 - xIntConv);
				data[offset].x += zInt * xIntConv;
				double temp = aInt;
				aInt = data[offset].a;
				data[offset].a = temp;
				size_t dn = s / pieces;
				d = max(0.0, data[s * size].a) / 2;
				zInt += z * d;
				aInt *= d;
				aInt += data[offset].a;
			}
			T u = (data[(sLeft + sCount) * size + r + 1].x - zInt) * (1.0 / aInt);
			data[n * pieces * size].z = u;
			for (size_t si = 1; si < sCount; ++si) {
				size_t offset = (sLeft + si) * size + r + 1;
				data[offset].x += u * (data[offset].a * xIntConv);
				double absx = abs(data[offset].x);
				if (absx > 1)
					data[offset].x *= 1.0 / absx;
			}
		}
	}
	for (size_t s = 0; s < segments * pieces * size; s += size)
	{
		const auto d = max(data[s].a, 0);
		for (size_t i = order + 1; i < size; ++i)
			data[s + i + size].x = data[s + i].x + data[s + i].z * d;
	}
}

template<size_t size, size_t pieces, size_t order, size_t uBackFree>
void Correct(Point * data) {
	constexpr size_t segments = (I1 << order) - 1;
	for (size_t r = 0; r < order; ++r) {
		size_t n0 = (I1 << r) - 1;
		for (size_t n = n0; n < segments; n += I1 << (r + 1)) {
			size_t sLeft = n * pieces;
			T mid = data[(sLeft + 1) * size + r].x + data[sLeft * size].z * midConv;
			double mod = abs(mid);
			if (mod > 1)
				mid *= (1.0 / mod);
			size_t off = r != 0 ? 1 : 0;
			for (size_t s = 1 - off; s < pieces + off; ++s)
				data[(sLeft + s) * size + r].x = mid;
			mod = min(max(mod, 1.0 / 2), 1.0 * 2);
			mod = (mod - 1) * dConv + 1;
			double d = data[sLeft * size].a * pieces;
			d += r;
			d *= mod;
			d = min(max(d, 1.0), 1.0 * 1000);
			d -= r;
			if (r == 1)
				d = max(d, d1Min);
			d /= pieces;
			for (size_t s = 0; s < pieces; ++s)
				data[(sLeft + s) * size].a = d;
		}
	}
	data[0].x = data[size].x;
	size_t back = segments * pieces * size;
	if (uBackFree)
		data[back].x = data[back - size].x;
	for (size_t k = uBackFree; k < order; ++k)
		(data[back + k].x *= (1 - xMidConv)) -= (data[back + k + 1].z * (xMidConv / data[back + k + 1].a));
}
