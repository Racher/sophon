#include <array>
#include <fstream>
#include <filesystem>
#include <future>

#include "solver.h"
#include "models.h"

std::vector<std::future<void>> sims;
std::filesystem::path path;

template<void model(Point*, double), size_t order, size_t size = order + 2, bool uBackFree = true, size_t pieces = 8>
struct Sim
{
	char filename[64];
	const double initDuration;
	const std::array<double, size> start;
	const double target;
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

	Sim(const char* filename, double initDuration, std::array<double, size> start, double target)
		: initDuration(initDuration)
		, start(start)
		, target(target)
	{
		strcpy_s(this->filename, filename);
		strcat_s(this->filename, ".doubles");
		sims.push_back(std::async(std::move(*this)));
	}

	void operator()() {
		constexpr auto segments = (I1 << order) - 1;
		constexpr auto pointCount = pieces * segments + 1;
		constexpr auto iters = 250000;
		constexpr auto simSaveDiv = iters / 50;
		constexpr auto updateDiv = 100;
		constexpr auto stepDiv = 100;
		constexpr auto planIter = 2048;

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

int main() {
	path = std::filesystem::current_path() / "simdata";
	std::filesystem::remove_all(path);
	std::filesystem::create_directories(path);
	
	Sim<O1, 1>("O1", 2, {}, 0.75);
	Sim<O2, 2>("O2", 24, {}, 1);
	Sim<O3, 3>("O3", 30, {}, 1);
	Sim<O4, 4>("O4", 60, {}, 1);
	Sim<O4z, 4>("O4z", 60, {}, 1);
	Sim<O4c, 4>("O4c", 30, {0,0,0,-0.35386002862997135 }, 0);
	Sim<DC, 2>("DC", 100, {}, 0.75);
	Sim<DCpp, 2, 5> ("DCpp", 220, {}, 0.75);
	Sim<Servo, 3>("Servo", 56, {}, 1);
	Sim<Rocket, 4, 6, false>("Rocket", 30, {}, 1);
	Sim<DCr, 2>("DCr", 100, {}, 0.75);
	Sim<DCl, 2>("DCl", 100, {0.015, 0.75}, 0.75);

	for (auto& sim : sims)
		sim.get();
	return 0;
}
