#include "tracker/association.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::size_t tracks{1024};
  std::size_t measurements{1024};
  int warmup{3};
  int repetitions{20};
  std::string output;
  bool require_cuda{false};
};

Options parse_options(int argc, char** argv) {
  Options options;
  auto next = [&](int& index, const char* flag) {
    if (++index >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
    return std::string(argv[index]);
  };
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--tracks") options.tracks = std::stoull(next(i, "--tracks"));
    else if (argument == "--measurements")
      options.measurements = std::stoull(next(i, "--measurements"));
    else if (argument == "--warmup") options.warmup = std::stoi(next(i, "--warmup"));
    else if (argument == "--repetitions")
      options.repetitions = std::stoi(next(i, "--repetitions"));
    else if (argument == "--output") options.output = next(i, "--output");
    else if (argument == "--require-cuda") options.require_cuda = true;
    else throw std::runtime_error("unknown option: " + argument);
  }
  if (options.tracks == 0 || options.measurements == 0 || options.warmup < 0 ||
      options.repetitions < 1)
    throw std::runtime_error("sizes and repetitions must be positive; warmup must be nonnegative");
  return options;
}

double maximum_error(const std::vector<double>& expected, const std::vector<double>& actual) {
  if (expected.size() != actual.size()) return INFINITY;
  double result = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i)
    result = std::max(result, std::abs(expected[i] - actual[i]));
  return result;
}

template <typename Callable>
double elapsed_milliseconds(Callable&& callable) {
  const auto begin = std::chrono::steady_clock::now();
  callable();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

void write_row(std::ostream& output, const std::string& method, const Options& options,
               int repetition, double milliseconds, double kernel_milliseconds,
               double maximum_difference) {
  output << method << ',' << options.tracks << ',' << options.measurements << ',' << repetition
         << ',' << std::setprecision(12) << milliseconds << ',' << kernel_milliseconds << ','
         << maximum_difference << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::mt19937_64 generator(20261011);
    std::normal_distribution<double> distribution(0.0, 1000.0);
    std::vector<tracker::TrackView> tracks(options.tracks);
    std::vector<tracker::Measurement> measurements(options.measurements);
    for (std::size_t i = 0; i < tracks.size(); ++i)
      tracks[i] = {static_cast<int>(i), distribution(generator), distribution(generator)};
    for (auto& measurement : measurements) {
      measurement.value1 = distribution(generator);
      measurement.value2 = distribution(generator);
    }

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!options.output.empty()) {
      file.open(options.output);
      if (!file) throw std::runtime_error("cannot open benchmark output: " + options.output);
      output = &file;
    }
    *output << "method,objects,measurements,repetition,elapsed_ms,kernel_ms,max_abs_error\n";

    std::vector<double> reference;
#if TRACKER_ENABLE_THREADS
    for (int iteration = -options.warmup; iteration < options.repetitions; ++iteration) {
      const double elapsed = elapsed_milliseconds(
          [&] { reference = tracker::cpu_squared_distances(tracks, measurements, false); });
      if (iteration >= 0)
        write_row(*output, "cpu-serial", options, iteration, elapsed, 0.0, 0.0);
    }

    for (int iteration = -options.warmup; iteration < options.repetitions; ++iteration) {
      std::vector<double> threaded;
      const double elapsed = elapsed_milliseconds(
          [&] { threaded = tracker::cpu_squared_distances(tracks, measurements, true); });
      const double difference = maximum_error(reference, threaded);
      if (difference != 0.0)
        throw std::runtime_error("threaded CPU result disagrees with serial CPU result");
      if (iteration >= 0)
        write_row(*output, "cpu-threaded", options, iteration, elapsed, 0.0, difference);
    }
#else
    std::cerr << "threaded CPU backend disabled at build time; threaded benchmark not run\n";
#endif

    if (!tracker::gpu_backend_available()) {
      std::cerr << "CUDA not available; GPU benchmark not run\n";
      return options.require_cuda ? 2 : 0;
    }
    std::cerr << "CUDA backend active; device=" << tracker::gpu_device_name() << '\n';
    for (int iteration = -options.warmup; iteration < options.repetitions; ++iteration) {
      tracker::GpuDistanceResult result;
      const double elapsed = elapsed_milliseconds(
          [&] { result = tracker::gpu_squared_distances_timed(tracks, measurements); });
      const double difference = maximum_error(reference, result.distances);
      const double scale = *std::max_element(reference.begin(), reference.end());
      if (!std::isfinite(difference) || difference > 1e-12 * std::max(1.0, scale))
        throw std::runtime_error("CUDA result disagrees with serial CPU result");
      if (iteration >= 0) {
        write_row(*output, "cuda-end-to-end", options, iteration, elapsed,
                  result.kernel_milliseconds, difference);
        write_row(*output, "cuda-kernel", options, iteration, result.kernel_milliseconds,
                  result.kernel_milliseconds, difference);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "association_benchmark: " << error.what() << '\n';
    return 1;
  }
}
