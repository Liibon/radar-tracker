#include "tracker/association.hpp"
#include "tracker/csv.hpp"
#include "tracker/kalman_filter.hpp"
#include "tracker/track_manager.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

void close(double actual, double expected, double tolerance, const std::string& message) {
  check(std::abs(actual - expected) <= tolerance,
        message + " (actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected) + ")");
}

tracker::Measurement cart(double timestamp, const std::string& sensor,
                          double x, double y, int truth = -1) {
  tracker::Measurement result;
  result.timestamp = timestamp;
  result.sensor_id = sensor;
  result.type = tracker::MeasurementType::Cartesian;
  result.value1 = x;
  result.value2 = y;
  result.true_target_id = truth;
  result.position_variance = 0.01;
  return result;
}

tracker::Measurement polar(double timestamp, const std::string& sensor,
                           double range, double bearing, double sx, double sy,
                           int truth = -1) {
  tracker::Measurement result;
  result.timestamp = timestamp;
  result.sensor_id = sensor;
  result.type = tracker::MeasurementType::Polar;
  result.value1 = range;
  result.value2 = bearing;
  result.sensor_x = sx;
  result.sensor_y = sy;
  result.range_variance = 0.01;
  result.bearing_variance = 1e-8;
  result.true_target_id = truth;
  return result;
}

void test_filter_prediction_and_cartesian_update() {
  tracker::KalmanFilter filter(0.0, 0.0, 1.0, 1.0, 0.0);
  filter.update_cartesian(10.0, -4.0, 0.01);
  check(filter.x() > 9.8 && filter.y() < -3.9, "Cartesian update approaches precise measurement");
  const double before_x = filter.x();
  filter.predict(2.0);
  close(filter.x(), before_x + 2.0 * filter.vx(), 1e-9, "constant velocity prediction");
  bool rejected = false;
  try { filter.predict(-1.0); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "negative prediction interval is rejected");
}

void test_sensor_relative_polar_geometry() {
  const auto observation = polar(0.0, "offset-radar", 5.0, 0.0, 100.0, -20.0);
  close(tracker::measurement_x(observation), 105.0, 1e-12, "polar x includes sensor origin");
  close(tracker::measurement_y(observation), -20.0, 1e-12, "polar y includes sensor origin");

  tracker::KalmanFilter filter(104.0, -19.0, 10.0, 1.0, 0.0);
  const double error_before = std::hypot(filter.x() - 105.0, filter.y() + 20.0);
  filter.update(observation);
  const double error_after = std::hypot(filter.x() - 105.0, filter.y() + 20.0);
  check(error_after < error_before, "offset-sensor EKF update moves toward world-space target");
}

void test_association_is_one_to_one_and_deterministic() {
  const std::vector<tracker::TrackView> tracks{{1, -1.0, 0.0}, {2, 1.0, 0.0}};
  const std::vector<tracker::Measurement> measurements{
      cart(0.0, "s", 0.0, 0.0), cart(0.0, "s", 1.1, 0.0)};
  const auto matches = tracker::nearest_neighbor_association(tracks, measurements, 5.0, false);
  check(matches.size() == 2, "association produces two one-to-one matches");
  check(matches[0].track_index == 1 && matches[0].measurement_index == 1,
        "nearest pair is selected first");
  check(matches[1].track_index == 0 && matches[1].measurement_index == 0,
        "remaining pair is assigned without reuse");
}

void test_lifecycle_and_multisensor_fusion() {
  tracker::TrackManager manager(20.0, 1, 2, false, false);
  check(manager.process_batch(0.0, {cart(0.0, "a", 10.0, 5.0, 7)}).empty(),
        "new track is tentative");

  // Two measurements at the same timestamp both update the state, but count as
  // one lifecycle scan rather than instantly confirming a tentative track.
  auto estimates = manager.process_batch(
      0.0, {cart(0.0, "a", 10.0, 5.0, 7), polar(0.0, "b", 5.0, 0.0, 5.0, 5.0, 7)});
  check(estimates.empty(), "same-time second sensor does not double-count confirmation hits");
  estimates = manager.process_batch(
      1.0, {cart(1.0, "a", 11.0, 5.0, 7), polar(1.0, "b", 6.0, 0.0, 5.0, 5.0, 7)});
  check(estimates.size() == 1, "next scan confirms fused track");
  close(estimates.front().x, 11.0, 0.25, "two sensors fuse in common world frame");

  manager.process_batch(2.0, {});
  check(manager.tracks().size() == 1, "track coasts through configured miss count");
  manager.process_batch(3.0, {});
  check(manager.tracks().empty(), "track is deleted after exceeding miss count");
}

void test_csv_optional_sensor_columns() {
  const std::string path = "tracker_test_measurements.csv";
  {
    std::ofstream output(path);
    output << "timestamp,sensor_id,measurement_id,type,value1,value2,true_target_id,"
              "sensor_x,sensor_y,range_std,bearing_std,position_std\n"
           << "1,b,9,polar,5,0,3,100,-20,2,0.1,7\n";
  }
  const auto rows = tracker::read_measurements_csv(path);
  std::remove(path.c_str());
  check(rows.size() == 1, "extended measurement CSV is readable");
  close(tracker::measurement_x(rows.front()), 105.0, 1e-12, "CSV sensor origin reaches geometry");
  close(rows.front().range_variance, 4.0, 1e-12, "CSV range std is squared into variance");
  close(rows.front().bearing_variance, 0.01, 1e-12, "CSV bearing std is squared into variance");
}

}  // namespace

int main() {
  test_filter_prediction_and_cartesian_update();
  test_sensor_relative_polar_geometry();
  test_association_is_one_to_one_and_deterministic();
  test_lifecycle_and_multisensor_fusion();
  test_csv_optional_sensor_columns();
  if (failures) std::cerr << failures << " test(s) failed\n";
  return failures == 0 ? 0 : 1;
}
