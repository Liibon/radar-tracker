# Radar Multi-Object Tracker

An end-to-end tracking capstone: Python generates deterministic target motion,
noisy radar detections, clutter, and plots; C++17 performs Kalman/EKF tracking,
nearest-neighbor association, track lifecycle management, two-sensor fusion,
threaded CPU association, and optional CUDA association.

## What is implemented

- Constant-velocity Kalman filter for Cartesian position measurements
- Extended Kalman filter for sensor-relative range/bearing measurements
- Greedy one-to-one nearest-neighbor association with distance gating
- Tentative track creation, confirmation, coasting, and deletion
- Multiple targets, Poisson clutter, missed detections, and two sensors
- Serial CPU, threaded CPU, and a real CUDA distance-matrix backend
- Seeded simulation, path/error plots, unit and integration tests
- Benchmark tooling that keeps unavailable CUDA results distinct from CPU data

The CUDA backend is optional at build time. A non-CUDA machine builds a stub
that reports CUDA as unavailable; it never labels CPU fallback as GPU work.

## Quick start

Requirements: CMake 3.18+, a C++17 compiler, and Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

.venv/bin/python -m python.runner \
  --tracker './build/radar_tracker --measurements {measurements} --output {estimates}'
```

The demo writes reproducible CSV files to `output/` and figures to
`output/plots/`:

- `paths.png`: truth, detections, clutter, sensor locations, and estimates
- `errors.png`: position and velocity error over time

Use `--cartesian --one-sensor --targets 1` for the first milestone, or keep the
defaults for polar EKF tracking with multiple targets, clutter, and two sensors.
Run `python -m python.runner --help` for every scenario option.

## C++ tracker CLI

```text
radar_tracker --measurements INPUT.csv --output ESTIMATES.csv
              [--gate METERS] [--max-misses N] [--confirm-hits N]
              [--serial] [--gpu]
```

The measurement CSV stores each sensor's pose and noise parameters on every
row, so files are self-contained and asynchronous sensors can be fused in a
common world frame. Its schema is:

```text
timestamp,sensor_id,measurement_id,type,value1,value2,true_target_id,
sensor_x,sensor_y,range_std,bearing_std,position_std
```

`sensor_x` and `sensor_y` are required, including for a sensor at `(0, 0)`.
Readers reject missing coordinates instead of silently changing polar geometry
by assuming an origin.

## CUDA and performance

Configure with `-DTRACKER_ENABLE_CUDA=ON` (the default). CMake compiles
`src/gpu_association.cu` only when it finds a CUDA compiler. `--gpu` requests
the CUDA cost-matrix path for sufficiently large batches; runtime output states
whether CUDA was active, merely available, or unavailable.

Probe a machine before claiming a GPU result:

```bash
python3 tests/probe_cuda.py
```

Build and run the benchmark, save every raw timing sample, and plot it:

```bash
./build/association_benchmark --tracks 1024 --measurements 1024 \
  --warmup 3 --repetitions 20 --output benchmarks/results/run.csv
.venv/bin/python python/plot_performance.py benchmarks/results/run.csv \
  --output benchmarks/results/performance.png
```

Add `--require-cuda` when a missing CUDA backend should fail the run. See
[`docs/BENCHMARK_RECORD.md`](docs/BENCHMARK_RECORD.md) for the measurement
protocol. A valid result compares equivalent serial CPU, parallel CPU, and
CUDA outputs on the same workload and reports median and p95 latency.

## Project map

```text
include/tracker/     public C++ interfaces and small matrix library
src/                 filter, association, lifecycle, CSV, CLI, CUDA backend
python/              simulator, pipeline runner, metrics, and plotting
tests/               deterministic C++/Python tests and CUDA environment probe
docs/REPORT.md       mathematical model, design, evaluation, and limitations
docs/TEST_PLAN.md    verification and benchmark acceptance criteria
```

This is an educational reference implementation, not a safety-critical radar
system. The greedy Euclidean association baseline is intentionally transparent;
dense scenes benefit from covariance-aware global assignment, JPDA, or MHT.
