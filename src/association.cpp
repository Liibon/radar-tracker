#include "tracker/association.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace tracker {
namespace {

std::vector<double> cpu_distances_impl(const std::vector<TrackView>& tracks,
                                       const std::vector<Measurement>& measurements,
                                       bool parallel) {
  const std::size_t count = tracks.size() * measurements.size();
  std::vector<double> distances(count);
  auto calculate = [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      const std::size_t ti = index / measurements.size();
      const std::size_t mi = index % measurements.size();
      const double dx = tracks[ti].x - measurement_x(measurements[mi]);
      const double dy = tracks[ti].y - measurement_y(measurements[mi]);
      distances[index] = dx * dx + dy * dy;
    }
  };

#if TRACKER_ENABLE_THREADS
  const unsigned hardware = std::thread::hardware_concurrency();
  if (parallel && hardware > 1 && count >= 4096) {
    const std::size_t workers = std::min<std::size_t>(hardware, count);
    std::vector<std::thread> threads;
    const std::size_t chunk = (count + workers - 1) / workers;
    for (std::size_t i = 0; i < workers; ++i) {
      const std::size_t begin = i * chunk;
      const std::size_t end = std::min(count, begin + chunk);
      if (begin < end) threads.emplace_back(calculate, begin, end);
    }
    for (auto& thread : threads) thread.join();
    return distances;
  }
#else
  (void)parallel;
#endif
  calculate(0, count);
  return distances;
}

}  // namespace

std::vector<double> cpu_squared_distances(const std::vector<TrackView>& tracks,
                                          const std::vector<Measurement>& measurements,
                                          bool parallel) {
  if (tracks.empty() || measurements.empty()) return {};
  if (tracks.size() > std::numeric_limits<std::size_t>::max() / measurements.size())
    throw std::overflow_error("association distance matrix size overflow");
  return cpu_distances_impl(tracks, measurements, parallel);
}

std::vector<Match> nearest_neighbor_association(
    const std::vector<TrackView>& tracks, const std::vector<Measurement>& measurements,
    double gate_distance, bool parallel, bool prefer_gpu) {
  if (!std::isfinite(gate_distance) || gate_distance < 0.0)
    throw std::invalid_argument("gate distance must be finite and non-negative");
  if (tracks.empty() || measurements.empty()) return {};
  std::vector<double> distances;
  if (prefer_gpu && gpu_backend_available() && tracks.size() * measurements.size() >= 4096)
    distances = gpu_squared_distances(tracks, measurements);
  else
    distances = cpu_squared_distances(tracks, measurements, parallel);

  struct Candidate { std::size_t ti, mi; double distance; };
  std::vector<Candidate> candidates;
  const double gate2 = gate_distance * gate_distance;
  for (std::size_t ti = 0; ti < tracks.size(); ++ti)
    for (std::size_t mi = 0; mi < measurements.size(); ++mi) {
      const double d = distances[ti * measurements.size() + mi];
      if (d <= gate2) candidates.push_back({ti, mi, d});
    }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.distance != b.distance) return a.distance < b.distance;
              if (a.ti != b.ti) return a.ti < b.ti;
              return a.mi < b.mi;
            });
  std::vector<bool> used_tracks(tracks.size()), used_measurements(measurements.size());
  std::vector<Match> matches;
  for (const auto& candidate : candidates) {
    if (!used_tracks[candidate.ti] && !used_measurements[candidate.mi]) {
      used_tracks[candidate.ti] = true;
      used_measurements[candidate.mi] = true;
      matches.push_back({candidate.ti, candidate.mi, candidate.distance});
    }
  }
  return matches;
}

}  // namespace tracker
