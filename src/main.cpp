#include "tracker/csv.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::string output;
  double gate{50.0};
  int max_misses{2};
  int confirm_hits{2};
  bool parallel{true};
  bool gpu{false};
};

void usage(const char* program) {
  std::cerr << "Usage: " << program << " --measurements measurements.csv --output estimates.csv [options]\n"
            << "Options: --gate METERS --max-misses N --confirm-hits N --serial --gpu\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* flag) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[i];
    };
    if (arg == "--input" || arg == "--measurements") options.input = value(arg.c_str());
    else if (arg == "--output") options.output = value("--output");
    else if (arg == "--gate") options.gate = std::stod(value("--gate"));
    else if (arg == "--max-misses") options.max_misses = std::stoi(value("--max-misses"));
    else if (arg == "--confirm-hits") options.confirm_hits = std::stoi(value("--confirm-hits"));
    else if (arg == "--serial") options.parallel = false;
    else if (arg == "--gpu") options.gpu = true;
    else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (options.input.empty() || options.output.empty()) throw std::runtime_error("--measurements and --output are required");
  if (options.gate <= 0 || options.max_misses < 0 || options.confirm_hits < 1)
    throw std::runtime_error("gate and track-management parameters must be positive");
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto measurements = tracker::read_measurements_csv(options.input);
    tracker::TrackManager manager(options.gate, options.max_misses, options.confirm_hits,
                                  options.parallel, options.gpu);
    std::vector<tracker::Estimate> all_estimates;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t begin = 0; begin < measurements.size();) {
      std::size_t end = begin + 1;
      while (end < measurements.size() && std::abs(measurements[end].timestamp - measurements[begin].timestamp) < 1e-9) ++end;
      std::vector<tracker::Measurement> batch(measurements.begin() + begin, measurements.begin() + end);
      auto estimates = manager.process_batch(measurements[begin].timestamp, batch);
      all_estimates.insert(all_estimates.end(), estimates.begin(), estimates.end());
      begin = end;
    }
    tracker::write_estimates_csv(options.output, all_estimates);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr << "processed " << measurements.size() << " measurements into " << all_estimates.size()
              << " estimates in " << elapsed << " s; CUDA="
              << (tracker::gpu_backend_available() ? (options.gpu ? "requested" : "available") : "unavailable") << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "radar_tracker: " << error.what() << '\n';
    usage(argv[0]);
    return 1;
  }
}
