#pragma once

#include <vector>

#include "tracker/association.hpp"
#include "tracker/kalman_filter.hpp"

namespace tracker {

struct Track {
  int id;
  KalmanFilter filter;
  double last_timestamp;
  int hits{1};
  int misses{0};
  bool confirmed{false};
  int last_true_target_id{-1};
  double last_hit_timestamp;
};

struct Estimate {
  double timestamp;
  int track_id;
  double x, y, vx, vy;
  int truth_target_id;
};

class TrackManager {
 public:
  explicit TrackManager(double gate_distance = 50.0, int max_misses = 2,
                        int confirmation_hits = 2, bool parallel = true,
                        bool prefer_gpu = false, bool prefer_opencl = false);
  std::vector<Estimate> process_batch(double timestamp, const std::vector<Measurement>& measurements);
  const std::vector<Track>& tracks() const { return tracks_; }

 private:
  void start_track(const Measurement& measurement);
  std::vector<Track> tracks_;
  int next_id_{1};
  double gate_distance_;
  int max_misses_;
  int confirmation_hits_;
  bool parallel_;
  bool prefer_gpu_;
  bool prefer_opencl_;
};

}  // namespace tracker
