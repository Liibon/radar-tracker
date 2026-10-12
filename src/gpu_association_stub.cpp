#include "tracker/association.hpp"

#include <stdexcept>

namespace tracker {

bool gpu_backend_available() { return false; }

std::string gpu_device_name() { return {}; }

GpuDistanceResult gpu_squared_distances_timed(const std::vector<TrackView>&,
                                              const std::vector<Measurement>&) {
  throw std::runtime_error("CUDA backend was not built");
}

std::vector<double> gpu_squared_distances(const std::vector<TrackView>&,
                                          const std::vector<Measurement>&) {
  throw std::runtime_error("CUDA backend was not built");
}

}  // namespace tracker
