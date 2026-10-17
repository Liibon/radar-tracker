#include "tracker/association.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tracker {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

__global__ void squared_distance_kernel(const double* track_xy, const double* measurement_xy,
                                        double* output, std::size_t track_count,
                                        std::size_t measurement_count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = track_count * measurement_count;
  if (index >= count) return;
  const std::size_t ti = index / measurement_count;
  const std::size_t mi = index % measurement_count;
  const double dx = track_xy[2 * ti] - measurement_xy[2 * mi];
  const double dy = track_xy[2 * ti + 1] - measurement_xy[2 * mi + 1];
  output[index] = dx * dx + dy * dy;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) {
    cuda_check(cudaMalloc(&pointer_, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void* get() const { return pointer_; }

 private:
  void* pointer_{nullptr};
};

class CudaEvent {
 public:
  CudaEvent() { cuda_check(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

}  // namespace

bool cuda_backend_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_device_name() {
  if (!cuda_backend_available()) return {};
  int device = 0;
  cuda_check(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  return properties.name;
}

CudaDistanceResult cuda_squared_distances_timed(
    const std::vector<TrackView>& tracks,
    const std::vector<Measurement>& measurements) {
  if (!cuda_backend_available()) throw std::runtime_error("no CUDA device is available");
  if (tracks.empty() || measurements.empty()) return {};
  if (tracks.size() > std::numeric_limits<std::size_t>::max() / measurements.size())
    throw std::overflow_error("association distance matrix size overflow");
  std::vector<double> track_xy(2 * tracks.size());
  std::vector<double> measurement_xy(2 * measurements.size());
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    track_xy[2 * i] = tracks[i].x; track_xy[2 * i + 1] = tracks[i].y;
  }
  for (std::size_t i = 0; i < measurements.size(); ++i) {
    measurement_xy[2 * i] = measurement_x(measurements[i]);
    measurement_xy[2 * i + 1] = measurement_y(measurements[i]);
  }
  const std::size_t count = tracks.size() * measurements.size();
  CudaDistanceResult result;
  result.distances.resize(count);
  DeviceBuffer device_tracks(track_xy.size() * sizeof(double));
  DeviceBuffer device_measurements(measurement_xy.size() * sizeof(double));
  DeviceBuffer device_output(result.distances.size() * sizeof(double));
  cuda_check(cudaMemcpy(device_tracks.get(), track_xy.data(), track_xy.size() * sizeof(double),
                        cudaMemcpyHostToDevice), "copy tracks");
  cuda_check(cudaMemcpy(device_measurements.get(), measurement_xy.data(),
                        measurement_xy.size() * sizeof(double), cudaMemcpyHostToDevice),
             "copy measurements");

  constexpr unsigned threads = 256;
  const std::size_t block_count = (count + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("association kernel grid is too large");
  CudaEvent started;
  CudaEvent stopped;
  cuda_check(cudaEventRecord(started.get()), "record kernel start");
  squared_distance_kernel<<<static_cast<unsigned>(block_count), threads>>>(
      static_cast<const double*>(device_tracks.get()),
      static_cast<const double*>(device_measurements.get()),
      static_cast<double*>(device_output.get()), tracks.size(), measurements.size());
  cuda_check(cudaGetLastError(), "launch association kernel");
  cuda_check(cudaEventRecord(stopped.get()), "record kernel stop");
  cuda_check(cudaEventSynchronize(stopped.get()), "synchronize association kernel");
  cuda_check(cudaEventElapsedTime(&result.kernel_milliseconds, started.get(), stopped.get()),
             "measure association kernel");
  cuda_check(cudaMemcpy(result.distances.data(), device_output.get(),
                        result.distances.size() * sizeof(double), cudaMemcpyDeviceToHost),
             "copy results");
  return result;
}

std::vector<double> cuda_squared_distances(const std::vector<TrackView>& tracks,
                                           const std::vector<Measurement>& measurements) {
  return cuda_squared_distances_timed(tracks, measurements).distances;
}

}  // namespace tracker
