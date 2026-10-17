#include "tracker/association.hpp"

#include <stdexcept>

namespace tracker {

bool opencl_backend_available() { return false; }
std::string opencl_device_name() { return {}; }

OpenClDistanceResult opencl_squared_distances_timed(
    const std::vector<TrackView>&, const std::vector<Measurement>&) {
  throw std::runtime_error("OpenCL backend was not built");
}

std::vector<double> opencl_squared_distances(
    const std::vector<TrackView>&, const std::vector<Measurement>&) {
  throw std::runtime_error("OpenCL backend was not built");
}

}  // namespace tracker
