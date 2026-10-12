#include "tracker/association.hpp"
#include "tracker/kalman_filter.hpp"
#include "tracker/track_manager.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

tracker::Measurement cart(double time, double x, double y, int id = 1) {
  return {time, "cart", id, tracker::MeasurementType::Cartesian, x, y, 7};
}

void test_cartesian_filter() {
  tracker::KalmanFilter filter(0.0, 0.0);
  for (int i = 1; i <= 12; ++i) {
    filter.predict(1.0);
    filter.update_cartesian(5.0 * i, -2.0 * i, 1.0);
  }
  require(std::abs(filter.x() - 60.0) < 1.0, "Cartesian KF position did not converge");
  require(std::abs(filter.vx() - 5.0) < 0.6, "Cartesian KF x velocity did not converge");
  require(std::abs(filter.vy() + 2.0) < 0.6, "Cartesian KF y velocity did not converge");
}

void test_polar_filter() {
  tracker::KalmanFilter filter(100.0, 10.0);
  const double before = std::hypot(filter.x() - 80.0, filter.y() - 80.0);
  const double range = std::hypot(80.0, 80.0);
  const double bearing = std::atan2(80.0, 80.0);
  for (int i = 0; i < 5; ++i) filter.update_polar(range, bearing, 1.0, 1e-4);
  const double after = std::hypot(filter.x() - 80.0, filter.y() - 80.0);
  require(after < before * 0.2, "polar EKF update did not reduce position error");
  require(std::abs(tracker::wrap_angle(3.5) + 2.78318530718) < 1e-9,
          "bearing normalization is incorrect");

  tracker::Measurement offset;
  offset.timestamp = 0.0;
  offset.sensor_id = "radar-2";
  offset.type = tracker::MeasurementType::Polar;
  offset.sensor_x = 1800.0;
  offset.sensor_y = -900.0;
  offset.value1 = std::hypot(10.0, 20.0);
  offset.value2 = std::atan2(20.0, 10.0);
  require(std::abs(tracker::measurement_x(offset) - 1810.0) < 1e-9,
          "offset polar measurement x was not converted to world coordinates");
  require(std::abs(tracker::measurement_y(offset) + 880.0) < 1e-9,
          "offset polar measurement y was not converted to world coordinates");
  tracker::KalmanFilter offset_filter(1808.0, -884.0);
  const double offset_before = std::hypot(offset_filter.x() - 1810.0,
                                          offset_filter.y() + 880.0);
  offset_filter.update(offset);
  const double offset_after = std::hypot(offset_filter.x() - 1810.0,
                                         offset_filter.y() + 880.0);
  require(offset_after < offset_before,
          "offset-sensor EKF update did not reduce world-coordinate error");
}

void test_association() {
  const std::vector<tracker::TrackView> tracks{{1, 0.0, 0.0}, {2, 100.0, 0.0}};
  const std::vector<tracker::Measurement> measurements{cart(0, 98.0, 1.0), cart(0, 2.0, -1.0, 2),
                                                       cart(0, 500.0, 500.0, 3)};
  const auto matches = tracker::nearest_neighbor_association(tracks, measurements, 10.0, false, false);
  require(matches.size() == 2, "association should produce two gated matches");
  for (const auto& match : matches) {
    if (match.track_index == 0) require(match.measurement_index == 1, "first track got wrong measurement");
    if (match.track_index == 1) require(match.measurement_index == 0, "second track got wrong measurement");
  }
}

void test_track_lifecycle_and_fusion() {
  tracker::TrackManager manager(20.0, 1, 2, false, false);
  require(manager.process_batch(0.0, {cart(0, 10, 0)}).empty(), "new track confirmed too early");

  tracker::Measurement radar{1.0, "radar", 2, tracker::MeasurementType::Polar,
                             std::hypot(12.0, 1.0), std::atan2(1.0, 12.0), 7};
  const auto estimates = manager.process_batch(1.0, {cart(1, 12, 1), radar});
  require(estimates.size() == 1, "same-timestamp sensors should fuse into one confirmed track");
  require(manager.tracks().size() == 1, "sensor fusion created a duplicate track");

  manager.process_batch(2.0, {});
  manager.process_batch(3.0, {});
  require(manager.tracks().empty(), "track was not dropped after exceeding max misses");
}

}  // namespace

int main() {
  try {
    test_cartesian_filter();
    test_polar_filter();
    test_association();
    test_track_lifecycle_and_fusion();
    std::cout << "all tracker core tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
