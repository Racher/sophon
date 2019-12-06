#include <array>
#include <fstream>
#include <filesystem>
#include <future>
#include <numeric>
#include <string>

#pragma warning( disable : 6294 )
#pragma warning( disable : 26444 )

using T = double;
constexpr auto xMidConv = 1.0;
constexpr auto xIntConv = 1.0;
constexpr auto midConv = 0.5;
constexpr auto dConv = 0.06;
constexpr auto d1Min = 0.0;
constexpr auto zeroT = 0.0;
constexpr auto oneI = size_t(1);
constexpr auto min(double a, double b) { return std::min(a, b); }
constexpr auto max(double a, double b) { return std::max(a, b); }
constexpr auto sgn(double a) { return (0 < a) - (a < 0); }

struct Point {
	T z = zeroT;
	double a = 1;
	T x = zeroT;
};

using Model = void(Point*, double);

std::vector<std::future<void>> sims;
std::filesystem::path path;

template<size_t order>
void SetBack(Point* back) {
	for (size_t k = 1; k < order; ++k)
		(back[k].x *= (1 - xMidConv)) -= (back[k + 1].z * (xMidConv / back[k + 1].a));
}

template<size_t size, size_t pieces, size_t order>
void Predict(Point* data) {
	constexpr size_t segments = (oneI << order) - 1;
	for (size_t r = 0; r < order - 1; ++r)
		for (size_t m = (oneI << r) - 1; m < segments / 2; m += oneI << (r + 1)) {
			size_t sLeft = (2 * m + 1) * pieces;
			double lam = -data[(sLeft + 1)*size].a * pieces;
			for (size_t k = 0; k <= r; ++k) {
				double klam = k != 0 ? min(lam - k + 1, 1.0) : 0;
				T avg = zeroT;
				if (klam > 0) {
					size_t dist = pieces * (oneI << k);
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
		size_t sCount = ((oneI << (r + 1)) - 1) * pieces;
		size_t n0 = (oneI << r) - 1;
		for (size_t n = n0; n < segments; n += oneI << (r + 1)) {
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

template<size_t size, size_t pieces, size_t order, bool uBackFree>
void Correct(Point* data) {
	size_t segments = (oneI << order) - 1;
	for (size_t r = 0; r < order; ++r) {
		size_t n0 = (oneI << r) - 1;
		for (size_t n = n0; n < segments; n += oneI << (r + 1)) {
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
	if (uBackFree) {
		size_t back = segments * pieces * size;
		data[back].x = data[back - size].x;
	}
}

template<Model model, size_t order, size_t size = order + 2, bool uBackFree = true, size_t pieces = 8>
struct Sim
{
	char filename[64];
	const double initDuration;
	const std::array<double, size> start;
	const double target;
	const size_t planIter;
	std::vector<Point> data;
	std::vector<double> planData;

	void ApplyModel()
	{
		for (size_t s = 0; s < data.size(); s += size) {
			Point* const line = &data[s];
			double t[order];
			constexpr double dux = 1.0 / 64;
			for (size_t i = 0; i < order; ++i) {
				line[i].x += dux;
				model(line, 0);
				t[i] = line[i + 1].z;
				line[i].x -= dux;
			}
			model(line, 0);
			for (size_t i = 0; i < order; ++i) {
				line[i + 1].a = max(1.0 / 10000, (t[i] - line[i + 1].z) / dux);
				line[i + 1].z -= line[i + 1].a * line[i].x;
			}
		}
	}

	void AppendBinary(double y)
	{
		for (size_t s = 0; s < data.size(); s += size)
			for (size_t i = 0; i < size; ++i)
				if (i != order + 1)
				{
					planData.push_back(data[s + order + 1].x);
					planData.push_back(y);
					planData.push_back(data[s + i].x);
					planData.push_back(data[s].a * pieces);
					planData.push_back(data[s].z);
				}
	}

	Sim(const char* filename, double initDuration, std::array<double, size> start, double target, size_t planIter = 512)
		: initDuration(initDuration)
		, start(start)
		, target(target)
		, planIter(planIter)
	{
		strcpy_s(this->filename, filename);
		strcat_s(this->filename, ".binary64");
		sims.push_back(std::async(std::move(*this)));
	}

	void operator()() {
		constexpr auto segments = (oneI << order) - 1;
		constexpr auto pointCount = pieces * segments + 1;
		constexpr auto iters = 250000;
		constexpr auto simSaveDiv = iters / 50;
		constexpr auto updateDiv = 100;
		constexpr auto stepDiv = 100;

		data.resize(pointCount * size);
		planData = { (double)pointCount, (double)(size - 1), 0 };

		std::array<Point, size> state;
		(data.end() - size + order)->x = target;
		for (size_t i = 1; i < size; ++i)
			data[i].x = state[i].x = start[i - 1];
		for (size_t s = 0; s < data.size(); s += size)
		{
			data[s + order + 1].x = (double)(s / size);
			data[s].a = initDuration / segments / pieces;
		}

		size_t planItTarget = 1;
		for (size_t iter = 0;; ++iter) {
			if (iter == planItTarget)
			{
				AppendBinary(std::log2(iter));
				planItTarget *= 2;
			}
			if (iter == planIter)
				break;
			Correct<size, pieces, order, uBackFree>(data.data());
			ApplyModel();
			SetBack<order>(&*(data.end() - size));
			Predict<size, pieces, order>(data.data());
		}
		planData[2] = (double)(planData.size() - 3) / planData[0] / planData[1] / 5;

		const double simDuration = (data.end() - size + order + 1)->x * 1.5;
		const double dt = simDuration / iters / stepDiv;

		for (size_t iter = 0;; ++iter)
		{
			if (iter % updateDiv == 0) {
				state[0].x = data[0].x;
				for (size_t i = 1; i < size; ++i)
					data[i].x = state[i].x;
			}

			if (iter % simSaveDiv == 0)
				AppendBinary(state[order + 1].x);

			if (iter == iters)
				break;

			Correct<size, pieces, order, uBackFree>(data.data());
			ApplyModel();
			SetBack<order>(&*(data.end() - size));
			Predict<size, pieces, order>(data.data());

			for (size_t j = 0; j < stepDiv; ++j) {
				model(state.data(), 1);
				for (size_t i = 1; i < size; ++i)
					state[i].x += state[i].z * dt;
			}
		}

		std::filesystem::path path(path);
		path.append(filename);
		std::ofstream ofs(path.c_str(), std::ofstream::binary);
		ofs.write(reinterpret_cast<const char*>(planData.data()), planData.size() * sizeof(double) / sizeof(char));
	}
};

void O1(Point* data, double rand) {
	data[1].z = data[0].x - data[1].x;
	data[2].z = 1;
}

void O2(Point* data, double rand) {
	data[1].z = 0.1 * ((2 + std::sin(data[3].x)) * data[0].x + data[2].x * data[2].x);
	data[2].z = 0.1 * (data[1].x * data[1].x);
	data[3].z = 1;
}

void O4(Point* data, double rand) {
	data[1].z = 0.2 * data[0].x;
	data[2].z = 0.1 * data[1].x;
	data[3].z = 0.05 * data[2].x;
	data[4].z = 0.025 * data[3].x;
	data[5].z = 1;
}

void O4z(Point* data, double rand) {
	data[1].z = 0.06 * data[0].x;
	data[2].z = 0.15 * data[1].x;
	data[3].z = 0.05 * data[2].x;
	data[4].z = 0.036 * data[3].x;
	data[5].z = 1;
}

void DC(Point* data, double rand) {
	double di = -data[2].x - data[1].x + data[0].x * 2.01; //slight change so that a point referred to gets plotted
	double dw = -data[2].x + data[1].x;
	constexpr double tRef = 0.01;
	data[1].z = tRef * di;
	data[2].z = tRef * dw;
	data[3].z = 1;
}

void DCpp(Point* data, double rand) {
	constexpr auto T = 300;
	constexpr auto tRef = 0.01;
	auto t = data[3].x;
	auto u = data[0].x + rand * (std::sin(t / T + 8946) / 128 + std::sin(t * 2 + 6842) / 128);
	auto i = data[1].x + rand * (std::sin(t / T + 1351) / 128 + std::sin(t * 2 + 8135) / 128);
	auto w = data[2].x + rand * (std::sin(t / T + 7643) / 128 + std::sin(t * 2 + 9423) / 128);
	auto heat = data[4].x;

	auto resistance = 0.75 + heat / 2;
	auto cooling = 1 + abs(w) / 2 + std::sin(t / T + 3175) / 4 * rand;
	auto mu_visc = (1 - heat / 4) * (1 + std::sin(t / T + 1501) / 10 * rand);
	auto mu_flat = 0.2 * (1 + std::sin(t / T + 4351) / 10 * rand);
	auto mu_stick = rand * 0.05 * (1 + std::sin(t / T + 2351) / 4);

	auto dw_fric = -sgn(w) * mu_flat - w * mu_visc - (abs(w) < 0.01 ? 0.0 : sgn(w) * mu_stick);
	auto dw_i = 1 + std::sin(t / T + 2915) * rand / 10;
	auto di_w = -1 * (1 + std::sin(t / T + 2151) / 6 * rand);
	auto di_u = 2 * (1 + std::sin(t / T + 1321) / 10 * rand);

	auto pwm = sgn(u - std::remainder(0.4 + t * 12, 2));
	(u *= (1 - rand)) += rand * pwm;
	auto dw = i * dw_i + dw_fric;
	auto di = w * di_w - i * resistance + u * di_u;
	auto dheat = i * i * resistance - heat * cooling;

	data[1].z = tRef * di;
	data[2].z = tRef * dw;
	data[3].z = 1;
	data[4].z = tRef * dheat;
}

void Servo(Point* data, double) {
	double di = -data[2].x - data[1].x + data[0].x * 2;
	double dw = -data[2].x + data[1].x;
	double da = data[2].x;
	constexpr double tRef = 0.01;
	data[1].z = tRef * di;
	data[2].z = tRef * dw;
	data[3].z = tRef * da;
	data[4].z = 1;
}

void Rocket(Point* data, double rand) {
	constexpr double t = 0.5;
	constexpr double lim = 0.3;
	data[1].z = t * 0.5 * (data[0].x + 0.5 * data[3].x * data[3].x * std::cos(data[2].x * lim));
	data[2].z = t * data[1].x;
	data[3].z = t * 0.25 * (std::tan((data[2].x - 0.5 * data[0].x) * lim) / lim + -0.2 * data[3].x * data[3].x);
	data[4].z = t * 0.16 * data[3].x;
	data[5].z = 1;
}

void O4Compare(Point* data, double rand) {
	data[1].z = 0.1 * data[0].x * (0.85 + rand * 0.15);
	data[2].z = 0.1 * data[1].x;
	data[3].z = 0.1 * data[2].x;
	data[4].z = 0.1 * data[3].x;
	data[5].z = 1;
}

int main() {
	path = std::filesystem::current_path() / "simdata";
	std::filesystem::remove_all(path);
	std::filesystem::create_directories(path);
	
	Sim<O1, 1>("1st order", 2, {}, 0.75);
	Sim<O2, 2>("2nd order", 24, {}, 1);
	Sim<O4, 4>("4th order", 60, {}, 1);
	Sim<O4z, 4>("4th order zero mids", 60, {}, 1);
	Sim<DC, 2>("DC", 100, {}, 0.75, 2048);
	Sim<DCpp, 2, 5> ("DCpp", 220, {}, 0.75, 2048);
	Sim<Servo, 3>("Servo", 56, {}, 1, 2048);
	Sim<Rocket, 4, 6, false>("Rocket", 30, {}, 1, 2048);
	Sim<O4Compare, 4>("4th order compare", 30, {0,0,0,-0.35386002862997135 }, 0, 2048);

	for (auto& sim : sims)
		sim.get();
	return 0;
}
