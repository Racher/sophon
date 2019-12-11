#pragma once
#include <numeric>
#include "point.h"

void O1(Point* data, double) {
	data[1].z = data[0].x - data[1].x;
	data[2].z = 1;
}

void O2(Point* data, double) {
	data[1].z = 0.1 * ((2 + std::sin(data[3].x)) * data[0].x + data[2].x * data[2].x);
	data[2].z = 0.1 * (data[1].x * data[1].x);
	data[3].z = 1;
}

void O3(Point * data, double) {
	data[1].z = 0.2 * data[0].x;
	data[2].z = 0.1 * data[1].x;
	data[3].z = 0.05 * data[2].x;
	data[4].z = 1;
}

void O4(Point * data, double) {
	data[1].z = 0.2 * data[0].x;
	data[2].z = 0.1 * data[1].x;
	data[3].z = 0.05 * data[2].x;
	data[4].z = 0.025 * data[3].x;
	data[5].z = 1;
}

void O4z(Point * data, double) {
	data[1].z = 0.06 * data[0].x;
	data[2].z = 0.15 * data[1].x;
	data[3].z = 0.05 * data[2].x;
	data[4].z = 0.036 * data[3].x;
	data[5].z = 1;
}

void O4c(Point* data, double sim) {
	data[1].z = 0.1 * data[0].x * (0.85 + sim * 0.15); //Underestimate model to avoid overshoot
	data[2].z = 0.1 * data[1].x;
	data[3].z = 0.1 * data[2].x;
	data[4].z = 0.1 * data[3].x;
	data[5].z = 1;
}

void DC(Point * data, double) {
	double di = -data[2].x - data[1].x + data[0].x * 2;
	double dw = -data[2].x + data[1].x;
	constexpr double tRef = 0.01;
	data[1].z = tRef * di;
	data[2].z = tRef * dw;
	data[3].z = 1;
}

void DCpp(Point * data, double sim) {
	constexpr auto T = 300;
	constexpr auto tRef = 0.01;
	auto t = data[3].x;
	auto u = data[0].x + sim * (std::sin(t / T + 8946) / 128 + std::sin(t * 2 + 6842) / 128);
	auto i = data[1].x + sim * (std::sin(t / T + 1351) / 128 + std::sin(t * 2 + 8135) / 128);
	auto w = data[2].x + sim * (std::sin(t / T + 7643) / 128 + std::sin(t * 2 + 9423) / 128);
	auto heat = data[4].x;

	auto resistance = 0.75 + heat / 2;
	auto cooling = 1 + abs(w) / 2 + std::sin(t / T + 3175) / 4 * sim;
	auto mu_visc = (1 - heat / 4) * (1 + std::sin(t / T + 1501) / 10 * sim);
	auto mu_flat = 0.2 * (1 + std::sin(t / T + 4351) / 10 * sim);
	auto mu_stick = sim * 0.05 * (1 + std::sin(t / T + 2351) / 4);

	auto dw_fric = -sgn(w) * mu_flat - w * mu_visc - (abs(w) < 0.01 ? 0.0 : sgn(w) * mu_stick);
	auto dw_i = 1 + std::sin(t / T + 2915) * sim / 10;
	auto di_w = -1 * (1 + std::sin(t / T + 2151) / 6 * sim);
	auto di_u = 2 * (1 + std::sin(t / T + 1321) / 10 * sim);

	auto pwm = sgn(u - std::remainder(t * 10, 2));
	(u *= (1 - sim)) += sim * pwm;
	auto dw = i * dw_i + dw_fric;
	auto di = w * di_w - i * resistance + u * di_u;
	auto dheat = i * i * resistance - heat * cooling;

	data[1].z = tRef * di;
	data[2].z = tRef * dw;
	data[3].z = 1;
	data[4].z = tRef * dheat;
}

void Servo(Point * data, double) {
	double di = -data[2].x - data[1].x + data[0].x * 2;
	double dw = -data[2].x + data[1].x;
	double da = data[2].x;
	constexpr double tRef = 0.01;
	data[1].z = tRef * di;
	data[2].z = tRef * dw;
	data[3].z = tRef * da;
	data[4].z = 1;
}

void Rocket(Point * data, double) {
	constexpr double t = 0.5;
	constexpr double lim = 0.3;
	data[1].z = t * 0.5 * (data[0].x + 0.5 * data[3].x * data[3].x * std::cos(data[2].x * lim));
	data[2].z = t * data[1].x;
	data[3].z = t * 0.25 * (std::tan((data[2].x - 0.5 * data[0].x) * lim) / lim + -0.2 * data[3].x * data[3].x);
	data[4].z = t * 0.16 * data[3].x;
	data[5].z = 1;
}
