#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tracker {

template <std::size_t R, std::size_t C>
class Matrix {
 public:
  constexpr Matrix() : data_{} {}

  double& operator()(std::size_t r, std::size_t c) { return data_[r * C + c]; }
  double operator()(std::size_t r, std::size_t c) const { return data_[r * C + c]; }

  static Matrix<R, C> zeros() { return {}; }

  static Matrix<R, C> identity() {
    static_assert(R == C, "identity requires a square matrix");
    Matrix<R, C> result;
    for (std::size_t i = 0; i < R; ++i) result(i, i) = 1.0;
    return result;
  }

  Matrix<R, C>& operator+=(const Matrix<R, C>& rhs) {
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += rhs.data_[i];
    return *this;
  }

  Matrix<R, C>& operator-=(const Matrix<R, C>& rhs) {
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= rhs.data_[i];
    return *this;
  }

 private:
  std::array<double, R * C> data_;
};

template <std::size_t R, std::size_t C>
Matrix<R, C> operator+(Matrix<R, C> lhs, const Matrix<R, C>& rhs) { return lhs += rhs; }

template <std::size_t R, std::size_t C>
Matrix<R, C> operator-(Matrix<R, C> lhs, const Matrix<R, C>& rhs) { return lhs -= rhs; }

template <std::size_t R, std::size_t C>
Matrix<R, C> operator*(const Matrix<R, C>& value, double scalar) {
  Matrix<R, C> result;
  for (std::size_t r = 0; r < R; ++r)
    for (std::size_t c = 0; c < C; ++c) result(r, c) = value(r, c) * scalar;
  return result;
}

template <std::size_t R, std::size_t K, std::size_t C>
Matrix<R, C> operator*(const Matrix<R, K>& lhs, const Matrix<K, C>& rhs) {
  Matrix<R, C> result;
  for (std::size_t r = 0; r < R; ++r)
    for (std::size_t c = 0; c < C; ++c)
      for (std::size_t k = 0; k < K; ++k) result(r, c) += lhs(r, k) * rhs(k, c);
  return result;
}

template <std::size_t R, std::size_t C>
Matrix<C, R> transpose(const Matrix<R, C>& value) {
  Matrix<C, R> result;
  for (std::size_t r = 0; r < R; ++r)
    for (std::size_t c = 0; c < C; ++c) result(c, r) = value(r, c);
  return result;
}

template <std::size_t N>
Matrix<N, N> inverse(const Matrix<N, N>& value) {
  Matrix<N, N> a = value;
  Matrix<N, N> inv = Matrix<N, N>::identity();
  for (std::size_t col = 0; col < N; ++col) {
    std::size_t pivot = col;
    for (std::size_t row = col + 1; row < N; ++row)
      if (std::abs(a(row, col)) > std::abs(a(pivot, col))) pivot = row;
    if (std::abs(a(pivot, col)) < 1e-12) throw std::runtime_error("singular matrix");
    if (pivot != col) {
      for (std::size_t c = 0; c < N; ++c) {
        const double av = a(col, c); a(col, c) = a(pivot, c); a(pivot, c) = av;
        const double iv = inv(col, c); inv(col, c) = inv(pivot, c); inv(pivot, c) = iv;
      }
    }
    const double scale = a(col, col);
    for (std::size_t c = 0; c < N; ++c) { a(col, c) /= scale; inv(col, c) /= scale; }
    for (std::size_t row = 0; row < N; ++row) {
      if (row == col) continue;
      const double factor = a(row, col);
      for (std::size_t c = 0; c < N; ++c) {
        a(row, c) -= factor * a(col, c);
        inv(row, c) -= factor * inv(col, c);
      }
    }
  }
  return inv;
}

using Vector2 = Matrix<2, 1>;
using Vector4 = Matrix<4, 1>;
using Matrix2 = Matrix<2, 2>;
using Matrix4 = Matrix<4, 4>;

}  // namespace tracker
