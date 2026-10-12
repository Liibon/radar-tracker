cmake_minimum_required(VERSION 3.18)
project(radar_tracker LANGUAGES CXX)

include(CTest)

option(TRACKER_ENABLE_CUDA "Build the CUDA association backend when available" ON)
option(TRACKER_ENABLE_THREADS "Enable parallel CPU association" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TRACKER_SOURCES
    src/kalman_filter.cpp
    src/association.cpp
    src/track_manager.cpp
    src/csv.cpp)

set(TRACKER_HAS_CUDA OFF)
if(TRACKER_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(TRACKER_HAS_CUDA ON)
    list(APPEND TRACKER_SOURCES src/gpu_association.cu)
  endif()
endif()

if(NOT TRACKER_HAS_CUDA)
  list(APPEND TRACKER_SOURCES src/gpu_association_stub.cpp)
endif()

add_library(tracker_core ${TRACKER_SOURCES})
target_include_directories(tracker_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(TRACKER_ENABLE_THREADS)
  find_package(Threads REQUIRED)
  target_link_libraries(tracker_core PRIVATE Threads::Threads)
endif()
target_compile_definitions(tracker_core PUBLIC
  TRACKER_HAS_CUDA=$<BOOL:${TRACKER_HAS_CUDA}>
  TRACKER_ENABLE_THREADS=$<BOOL:${TRACKER_ENABLE_THREADS}>)

if(TRACKER_HAS_CUDA)
  set_target_properties(tracker_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(MSVC)
  target_compile_options(tracker_core PRIVATE /W4)
else()
  target_compile_options(tracker_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

add_executable(radar_tracker src/main.cpp)
target_link_libraries(radar_tracker PRIVATE tracker_core)

add_executable(association_benchmark benchmarks/association_benchmark.cpp)
target_link_libraries(association_benchmark PRIVATE tracker_core)

if(BUILD_TESTING)
  add_test(NAME association_benchmark_cpu_smoke
           COMMAND association_benchmark --tracks 16 --measurements 20
                   --warmup 1 --repetitions 2)
endif()

if(BUILD_TESTING)
  add_executable(tracker_sensor_tests tests/tracker_core_tests.cpp)
  target_link_libraries(tracker_sensor_tests PRIVATE tracker_core)
  add_test(NAME tracker_sensor_tests COMMAND tracker_sensor_tests)
endif()

if(BUILD_TESTING)
  add_executable(tracker_core_tests tests/test_tracker_core.cpp)
  target_link_libraries(tracker_core_tests PRIVATE tracker_core)
  add_test(NAME tracker_core_tests COMMAND tracker_core_tests)
endif()

message(STATUS "CUDA association backend: ${TRACKER_HAS_CUDA}")
message(STATUS "Parallel CPU association: ${TRACKER_ENABLE_THREADS}")
