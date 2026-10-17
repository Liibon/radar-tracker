#include "tracker/association.hpp"

#include <stdexcept>

namespace tracker {

bool cuda_backend_available() { return false; }

std::string cuda_device_name() { return {}; }

CudaDistanceResult cuda_squared_distances_timed(const std::vector<TrackView>&,
                                                 const std::vector<Measurement>&) {
  throw std::runtime_error("CUDA backend was not built");
}

std::vector<double> cuda_squared_distances(const std::vector<TrackView>&,
                                           const std::vector<Measurement>&) {
  throw std::runtime_error("CUDA backend was not built");
}

}  // namespace tracker
