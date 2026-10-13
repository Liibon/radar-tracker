# Radar Multi-Object Tracker

An end-to-end tracking capstone: Python imports real automotive radar or generates
deterministic synthetic scenarios and plots; C++17 performs Kalman/EKF tracking,
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
- A label-free RadarScenes adapter and byte-range sample downloader
- Benchmark tooling that keeps unavailable CUDA results distinct from CPU data

The CUDA backend is optional at build time. A non-CUDA machine builds a stub
that reports CUDA as unavailable; it never labels CPU fallback as GPU work.

## Real-world quick start

Requirements: CMake 3.18+, a C++17 compiler, and Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

# Fetch only sequence 158 (~8 MB extracted), not the complete 11 GB archive.
.venv/bin/python -m python.download_radarscenes_sample
.venv/bin/python -m python.run_radarscenes \
  real_data/radarscenes_sequence_158
```

This uses genuine measurements from four automotive radars in the
[RadarScenes dataset](https://radar-scenes.com/) (sequence 158). Input objects
are created with compensated-radial-velocity filtering and Euclidean connected
components; labels are not used by detection, clustering, association, or the
filter. Point-wise track labels are read separately for evaluation, where
tracks are matched independently by geometry instead of trusting labels copied
through tracker state. Outputs go
to `real_output/`, including `metrics.json`, trajectories, and position-error
plots. The downloaded data remains under RadarScenes' CC BY-NC-SA 4.0 license
and is intentionally excluded from Git.

See [`docs/REAL_DATA.md`](docs/REAL_DATA.md) for provenance, transformation
details, evaluation semantics, and limitations.

## Synthetic regression scenario

The simulator remains useful for exact, seeded tests where motion and noise are
known:

```bash
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

Newly generated files always include `sensor_x` and `sensor_y`, including for a
sensor at `(0, 0)`. The C++ reader retains origin defaults only for legacy
single-sensor files that predate the explicit coordinate columns.

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
docs/REAL_DATA.md    real-data provenance, processing, metrics, and caveats
docs/REPORT.md       mathematical model, design, evaluation, and limitations
docs/TEST_PLAN.md    verification and benchmark acceptance criteria
```

This is an educational reference implementation, not a safety-critical radar
system. The greedy Euclidean association baseline is intentionally transparent;
dense scenes benefit from covariance-aware global assignment, JPDA, or MHT.
