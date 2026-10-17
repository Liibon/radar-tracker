#include "tracker/association.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace tracker {
namespace {

void check(cl_int status, const char* operation) {
  if (status != CL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed (OpenCL error " +
                             std::to_string(status) + ")");
}

const char* kernel_source = R"CLC(
__kernel void squared_distances(__global const float* tracks,
                                __global const float* measurements,
                                __global float* output,
                                const ulong measurement_count,
                                const ulong pair_count) {
  const ulong index = get_global_id(0);
  if (index >= pair_count) return;
  const ulong track = index / measurement_count;
  const ulong measurement = index - track * measurement_count;
  const float dx = tracks[2 * track] - measurements[2 * measurement];
  const float dy = tracks[2 * track + 1] - measurements[2 * measurement + 1];
  output[index] = dx * dx + dy * dy;
}
)CLC";

class OpenClRuntime {
 public:
  OpenClRuntime() {
    cl_uint platform_count = 0;
    check(clGetPlatformIDs(0, nullptr, &platform_count), "query platforms");
    if (platform_count == 0) throw std::runtime_error("no OpenCL platform found");
    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "list platforms");

    const cl_device_type preferences[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR,
                                          CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_ALL};
    for (cl_device_type type : preferences) {
      for (cl_platform_id platform : platforms) {
        cl_uint count = 0;
        const cl_int status = clGetDeviceIDs(platform, type, 1, &device_, &count);
        if (status == CL_SUCCESS && count > 0) break;
      }
      if (device_) break;
    }
    if (!device_) throw std::runtime_error("no OpenCL device found");

    cl_int status = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &status);
    check(status, "create context");
    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &status);
    check(status, "create command queue");
    const std::size_t source_length = std::char_traits<char>::length(kernel_source);
    program_ = clCreateProgramWithSource(context_, 1, &kernel_source, &source_length, &status);
    check(status, "create program");
    status = clBuildProgram(program_, 1, &device_, "-cl-std=CL1.2", nullptr, nullptr);
    if (status != CL_SUCCESS) {
      std::size_t size = 0;
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
      std::string log(size, '\0');
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
      throw std::runtime_error("build OpenCL kernel failed: " + log);
    }
    kernel_ = clCreateKernel(program_, "squared_distances", &status);
    check(status, "create kernel");
  }

  ~OpenClRuntime() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
  }

  std::string device_name() const {
    std::size_t size = 0;
    check(clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &size), "query device name");
    std::string name(size, '\0');
    check(clGetDeviceInfo(device_, CL_DEVICE_NAME, size, name.data(), nullptr),
          "read device name");
    if (!name.empty() && name.back() == '\0') name.pop_back();
    return name;
  }

  OpenClDistanceResult run(const std::vector<TrackView>& tracks,
                           const std::vector<Measurement>& measurements) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracks.empty() || measurements.empty()) return {};
    if (tracks.size() > std::numeric_limits<std::size_t>::max() / measurements.size())
      throw std::overflow_error("association distance matrix size overflow");
    const std::size_t pair_count = tracks.size() * measurements.size();
    std::vector<float> track_xy(2 * tracks.size());
    std::vector<float> measurement_xy(2 * measurements.size());
    for (std::size_t i = 0; i < tracks.size(); ++i) {
      track_xy[2 * i] = static_cast<float>(tracks[i].x);
      track_xy[2 * i + 1] = static_cast<float>(tracks[i].y);
    }
    for (std::size_t i = 0; i < measurements.size(); ++i) {
      measurement_xy[2 * i] = static_cast<float>(measurement_x(measurements[i]));
      measurement_xy[2 * i + 1] = static_cast<float>(measurement_y(measurements[i]));
    }

    cl_int status = CL_SUCCESS;
    cl_mem track_buffer = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         track_xy.size() * sizeof(float), track_xy.data(), &status);
    check(status, "create track buffer");
    cl_mem measurement_buffer = clCreateBuffer(
        context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        measurement_xy.size() * sizeof(float), measurement_xy.data(), &status);
    if (status != CL_SUCCESS) {
      clReleaseMemObject(track_buffer);
      check(status, "create measurement buffer");
    }
    cl_mem output_buffer = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                          pair_count * sizeof(float), nullptr, &status);
    if (status != CL_SUCCESS) {
      clReleaseMemObject(measurement_buffer);
      clReleaseMemObject(track_buffer);
      check(status, "create output buffer");
    }

    try {
      const cl_ulong measurement_count_arg = measurements.size();
      const cl_ulong pair_count_arg = pair_count;
      check(clSetKernelArg(kernel_, 0, sizeof(track_buffer), &track_buffer), "set tracks");
      check(clSetKernelArg(kernel_, 1, sizeof(measurement_buffer), &measurement_buffer),
            "set measurements");
      check(clSetKernelArg(kernel_, 2, sizeof(output_buffer), &output_buffer), "set output");
      check(clSetKernelArg(kernel_, 3, sizeof(measurement_count_arg), &measurement_count_arg),
            "set measurement count");
      check(clSetKernelArg(kernel_, 4, sizeof(pair_count_arg), &pair_count_arg),
            "set pair count");
      cl_event event = nullptr;
      check(clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &pair_count, nullptr, 0, nullptr,
                                   &event),
            "launch kernel");
      check(clWaitForEvents(1, &event), "wait for kernel");
      std::vector<float> output(pair_count);
      check(clEnqueueReadBuffer(queue_, output_buffer, CL_TRUE, 0, output.size() * sizeof(float),
                                output.data(), 0, nullptr, nullptr),
            "read output");
      cl_ulong started = 0, stopped = 0;
      check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(started), &started,
                                    nullptr),
            "profile kernel start");
      check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(stopped), &stopped,
                                    nullptr),
            "profile kernel end");
      clReleaseEvent(event);
      clReleaseMemObject(output_buffer);
      clReleaseMemObject(measurement_buffer);
      clReleaseMemObject(track_buffer);
      OpenClDistanceResult result;
      result.distances.assign(output.begin(), output.end());
      result.kernel_milliseconds = static_cast<double>(stopped - started) / 1.0e6;
      return result;
    } catch (...) {
      clReleaseMemObject(output_buffer);
      clReleaseMemObject(measurement_buffer);
      clReleaseMemObject(track_buffer);
      throw;
    }
  }

 private:
  cl_device_id device_{nullptr};
  cl_context context_{nullptr};
  cl_command_queue queue_{nullptr};
  cl_program program_{nullptr};
  cl_kernel kernel_{nullptr};
  std::mutex mutex_;
};

OpenClRuntime& runtime() {
  static OpenClRuntime instance;
  return instance;
}

}  // namespace

bool opencl_backend_available() {
  try {
    (void)runtime();
    return true;
  } catch (...) {
    return false;
  }
}

std::string opencl_device_name() {
  try {
    return runtime().device_name();
  } catch (...) {
    return {};
  }
}

OpenClDistanceResult opencl_squared_distances_timed(
    const std::vector<TrackView>& tracks, const std::vector<Measurement>& measurements) {
  return runtime().run(tracks, measurements);
}

std::vector<double> opencl_squared_distances(
    const std::vector<TrackView>& tracks, const std::vector<Measurement>& measurements) {
  return opencl_squared_distances_timed(tracks, measurements).distances;
}

}  // namespace tracker
