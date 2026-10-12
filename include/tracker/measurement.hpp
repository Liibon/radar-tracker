#pragma once

#include <cmath>
#include <string>

namespace tracker {

enum class MeasurementType { Cartesian, Polar };

struct Measurement {
  double timestamp{};
  std::string sensor_id;
  int measurement_id{};
  MeasurementType type{MeasurementType::Cartesian};
  double value1{};
  double value2{};
  int true_target_id{-1};
  // Sensor pose in the world frame. Cartesian measurements are already in the
  // world frame; polar measurements are relative to this origin.
  double sensor_x{};
  double sensor_y{};
  double position_variance{9.0};
  double range_variance{16.0};
  double bearing_variance{0.0025};
};

inline double measurement_x(const Measurement& m) {
  return m.type == MeasurementType::Cartesian
             ? m.value1
             : m.sensor_x + m.value1 * std::cos(m.value2);
}

inline double measurement_y(const Measurement& m) {
  return m.type == MeasurementType::Cartesian
             ? m.value2
             : m.sensor_y + m.value1 * std::sin(m.value2);
}

}  // namespace tracker
