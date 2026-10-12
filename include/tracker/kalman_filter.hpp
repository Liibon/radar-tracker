#pragma once

#include "tracker/matrix.hpp"
#include "tracker/measurement.hpp"

namespace tracker {

class KalmanFilter {
 public:
  KalmanFilter(double x, double y, double position_variance = 100.0,
               double velocity_variance = 400.0, double process_acceleration_std = 2.0);

  void predict(double dt);
  void update_cartesian(double x, double y, double variance = 9.0);
  void update_polar(double range, double bearing, double range_variance = 16.0,
                    double bearing_variance = 0.0025, double sensor_x = 0.0,
                    double sensor_y = 0.0);
  void update(const Measurement& measurement);

  const Vector4& state() const { return state_; }
  const Matrix4& covariance() const { return covariance_; }
  double x() const { return state_(0, 0); }
  double y() const { return state_(1, 0); }
  double vx() const { return state_(2, 0); }
  double vy() const { return state_(3, 0); }

 private:
  void joseph_update(const Matrix<2, 4>& h, const Vector2& innovation, const Matrix2& r);
  Vector4 state_;
  Matrix4 covariance_;
  double process_acceleration_std_;
};

double wrap_angle(double radians);

}  // namespace tracker
