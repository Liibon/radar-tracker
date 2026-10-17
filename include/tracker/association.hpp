#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tracker/measurement.hpp"

namespace tracker {

struct TrackView { int id; double x; double y; };
struct Match { std::size_t track_index; std::size_t measurement_index; double squared_distance; };

struct CudaDistanceResult {
  std::vector<double> distances;
  // CUDA-event time for the kernel only. Allocation and transfers are excluded.
  float kernel_milliseconds{0.0F};
};

struct OpenClDistanceResult {
  std::vector<double> distances;
  // Device-event time for the portable float32 kernel only.
  double kernel_milliseconds{0.0};
};

std::vector<Match> nearest_neighbor_association(
    const std::vector<TrackView>& tracks, const std::vector<Measurement>& measurements,
    double gate_distance, bool parallel = true, bool prefer_gpu = false,
    bool prefer_opencl = false);

std::vector<double> cpu_squared_distances(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements,
    bool parallel = true);
bool cuda_backend_available();
std::string cuda_device_name();
CudaDistanceResult cuda_squared_distances_timed(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements);
std::vector<double> cuda_squared_distances(const std::vector<TrackView>& tracks,
                                           const std::vector<Measurement>& measurements);
bool opencl_backend_available();
std::string opencl_device_name();
OpenClDistanceResult opencl_squared_distances_timed(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements);
std::vector<double> opencl_squared_distances(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements);

}  // namespace tracker
