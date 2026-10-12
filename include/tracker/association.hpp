#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tracker/measurement.hpp"

namespace tracker {

struct TrackView { int id; double x; double y; };
struct Match { std::size_t track_index; std::size_t measurement_index; double squared_distance; };

struct GpuDistanceResult {
  std::vector<double> distances;
  // CUDA-event time for the kernel only. Allocation and transfers are excluded.
  float kernel_milliseconds{0.0F};
};

std::vector<Match> nearest_neighbor_association(
    const std::vector<TrackView>& tracks, const std::vector<Measurement>& measurements,
    double gate_distance, bool parallel = true, bool prefer_gpu = false);

std::vector<double> cpu_squared_distances(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements,
    bool parallel = true);
bool gpu_backend_available();
std::string gpu_device_name();
GpuDistanceResult gpu_squared_distances_timed(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements);
std::vector<double> gpu_squared_distances(const std::vector<TrackView>& tracks,
                                          const std::vector<Measurement>& measurements);

}  // namespace tracker
