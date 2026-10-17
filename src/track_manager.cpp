#include "tracker/track_manager.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tracker {

TrackManager::TrackManager(double gate_distance, int max_misses, int confirmation_hits,
                           bool parallel, bool prefer_gpu, bool prefer_opencl)
    : gate_distance_(gate_distance), max_misses_(max_misses),
      confirmation_hits_(confirmation_hits), parallel_(parallel), prefer_gpu_(prefer_gpu),
      prefer_opencl_(prefer_opencl) {
  if (!std::isfinite(gate_distance_) || gate_distance_ <= 0.0 || max_misses_ < 0 ||
      confirmation_hits_ < 1)
    throw std::invalid_argument("invalid track manager configuration");
}

void TrackManager::start_track(const Measurement& measurement) {
  Track track{next_id_++, KalmanFilter(measurement_x(measurement), measurement_y(measurement)),
              measurement.timestamp, 1, 0, false, -1, measurement.timestamp};
  track.confirmed = confirmation_hits_ <= 1;
  track.last_true_target_id = measurement.true_target_id;
  tracks_.push_back(std::move(track));
}

std::vector<Estimate> TrackManager::process_batch(
    double timestamp, const std::vector<Measurement>& measurements) {
  if (!std::isfinite(timestamp)) throw std::invalid_argument("batch timestamp must be finite");
  for (const auto& measurement : measurements) {
    if (!std::isfinite(measurement.timestamp) ||
        std::abs(measurement.timestamp - timestamp) > 1e-9)
      throw std::invalid_argument("all measurements in a batch must share its timestamp");
  }
  std::vector<int> existing_track_ids;
  existing_track_ids.reserve(tracks_.size());
  for (auto& track : tracks_) {
    if (timestamp + 1e-9 < track.last_timestamp)
      throw std::invalid_argument("measurement batches must be chronological");
    existing_track_ids.push_back(track.id);
    track.filter.predict(timestamp - track.last_timestamp);
    track.last_timestamp = timestamp;
  }

  // Associate one sensor at a time. This lets independent sensors update the same
  // track at one timestamp instead of competing for its single assignment.
  std::vector<std::string> sensor_order;
  std::unordered_map<std::string, std::vector<Measurement>> by_sensor;
  for (const auto& measurement : measurements) {
    if (!by_sensor.count(measurement.sensor_id)) sensor_order.push_back(measurement.sensor_id);
    by_sensor[measurement.sensor_id].push_back(measurement);
  }
  std::unordered_set<int> updated_track_ids;
  for (const auto& sensor_id : sensor_order) {
    const auto& sensor_measurements = by_sensor.at(sensor_id);
    std::vector<TrackView> views;
    views.reserve(tracks_.size());
    for (const auto& track : tracks_) views.push_back({track.id, track.filter.x(), track.filter.y()});
    const auto matches = nearest_neighbor_association(
        views, sensor_measurements, gate_distance_, parallel_, prefer_gpu_, prefer_opencl_);
    std::vector<bool> matched_measurements(sensor_measurements.size());
    for (const auto& match : matches) {
      Track& track = tracks_[match.track_index];
      const Measurement& measurement = sensor_measurements[match.measurement_index];
      track.filter.update(measurement);
      // Multiple independent sensors may update one track at the same time,
      // but lifecycle hits count scans, not raw measurements.
      if (timestamp > track.last_hit_timestamp + 1e-9) {
        track.hits++;
        track.last_hit_timestamp = timestamp;
      }
      track.misses = 0;
      track.confirmed = track.hits >= confirmation_hits_;
      if (measurement.true_target_id >= 0) track.last_true_target_id = measurement.true_target_id;
      updated_track_ids.insert(track.id);
      matched_measurements[match.measurement_index] = true;
    }
    for (std::size_t i = 0; i < sensor_measurements.size(); ++i)
      if (!matched_measurements[i]) start_track(sensor_measurements[i]);
  }
  for (auto& track : tracks_)
    if (std::find(existing_track_ids.begin(), existing_track_ids.end(), track.id) != existing_track_ids.end() &&
        !updated_track_ids.count(track.id)) {
      track.misses++;
    }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) { return t.misses > max_misses_; }),
                tracks_.end());

  std::vector<Estimate> estimates;
  for (const auto& track : tracks_) {
    if (track.confirmed)
      estimates.push_back({timestamp, track.id, track.filter.x(), track.filter.y(),
                           track.filter.vx(), track.filter.vy(), track.last_true_target_id});
  }
  return estimates;
}

}  // namespace tracker
