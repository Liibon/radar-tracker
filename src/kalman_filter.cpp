#include "tracker/kalman_filter.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tracker {
namespace {
constexpr double kPi = 3.14159265358979323846;
}

double wrap_angle(double radians) {
  if (!std::isfinite(radians)) return std::numeric_limits<double>::quiet_NaN();
  // remainder avoids an unbounded loop for very large, but finite, angles.
  radians = std::remainder(radians, 2.0 * kPi);
  while (radians > kPi) radians -= 2.0 * kPi;
  while (radians < -kPi) radians += 2.0 * kPi;
  return radians;
}

KalmanFilter::KalmanFilter(double x, double y, double position_variance,
                           double velocity_variance, double process_acceleration_std)
    : process_acceleration_std_(process_acceleration_std) {
  if (!std::isfinite(x) || !std::isfinite(y) || !(position_variance > 0.0) ||
      !(velocity_variance > 0.0) || !(process_acceleration_std >= 0.0))
    throw std::invalid_argument("invalid Kalman filter initial value or variance");
  state_(0, 0) = x;
  state_(1, 0) = y;
  covariance_(0, 0) = position_variance;
  covariance_(1, 1) = position_variance;
  covariance_(2, 2) = velocity_variance;
  covariance_(3, 3) = velocity_variance;
}

void KalmanFilter::predict(double dt) {
  if (!std::isfinite(dt) || dt < 0.0)
    throw std::invalid_argument("prediction dt must be finite and non-negative");
  if (dt == 0.0) return;
  Matrix4 f = Matrix4::identity();
  f(0, 2) = dt;
  f(1, 3) = dt;

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  const double q = process_acceleration_std_ * process_acceleration_std_;
  Matrix4 process;
  process(0, 0) = process(1, 1) = 0.25 * dt4 * q;
  process(0, 2) = process(2, 0) = 0.5 * dt3 * q;
  process(1, 3) = process(3, 1) = 0.5 * dt3 * q;
  process(2, 2) = process(3, 3) = dt2 * q;

  state_ = f * state_;
  covariance_ = f * covariance_ * transpose(f) + process;
}

void KalmanFilter::joseph_update(const Matrix<2, 4>& h, const Vector2& innovation,
                                 const Matrix2& r) {
  const Matrix2 s = h * covariance_ * transpose(h) + r;
  const Matrix<4, 2> gain = covariance_ * transpose(h) * inverse(s);
  state_ += gain * innovation;
  const Matrix4 i_minus_kh = Matrix4::identity() - gain * h;
  covariance_ = i_minus_kh * covariance_ * transpose(i_minus_kh) + gain * r * transpose(gain);
}

void KalmanFilter::update_cartesian(double x, double y, double variance) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(variance) || variance < 0.0)
    throw std::invalid_argument("invalid Cartesian measurement or variance");
  Matrix<2, 4> h;
  h(0, 0) = 1.0;
  h(1, 1) = 1.0;
  Vector2 z;
  z(0, 0) = x;
  z(1, 0) = y;
  Matrix2 r;
  r(0, 0) = r(1, 1) = variance;
  joseph_update(h, z - h * state_, r);
}

void KalmanFilter::update_polar(double range, double bearing, double range_variance,
                                double bearing_variance, double sensor_x, double sensor_y) {
  if (!std::isfinite(range) || range < 0.0 || !std::isfinite(bearing) ||
      !std::isfinite(sensor_x) || !std::isfinite(sensor_y) ||
      !std::isfinite(range_variance) || range_variance < 0.0 ||
      !std::isfinite(bearing_variance) || bearing_variance < 0.0)
    throw std::invalid_argument("invalid polar measurement, sensor pose, or variance");
  const double px = state_(0, 0) - sensor_x;
  const double py = state_(1, 0) - sensor_y;
  const double rho2 = px * px + py * py;
  const double safe_rho2 = rho2 < 1e-8 ? 1e-8 : rho2;
  const double rho = std::sqrt(safe_rho2);

  Matrix<2, 4> h;
  h(0, 0) = px / rho;
  h(0, 1) = py / rho;
  h(1, 0) = -py / safe_rho2;
  h(1, 1) = px / safe_rho2;
  Vector2 innovation;
  innovation(0, 0) = range - rho;
  innovation(1, 0) = wrap_angle(bearing - std::atan2(py, px));
  Matrix2 r;
  r(0, 0) = range_variance;
  r(1, 1) = bearing_variance;
  joseph_update(h, innovation, r);
}

void KalmanFilter::update(const Measurement& measurement) {
  if (measurement.type == MeasurementType::Cartesian)
    update_cartesian(measurement.value1, measurement.value2, measurement.position_variance);
  else
    update_polar(measurement.value1, measurement.value2, measurement.range_variance,
                 measurement.bearing_variance, measurement.sensor_x, measurement.sensor_y);
}

}  // namespace tracker
