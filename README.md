# Radar Tracker

A C++ multi-object radar tracker tested with real automotive radar data.

It includes:

- Kalman and extended Kalman filters
- Multiple-target tracking and clutter handling
- Two-sensor fusion
- CPU multithreading and CUDA acceleration
- Python tools for downloading data, running tests, and plotting results

## Run it

Requirements: CMake, a C++17 compiler, and Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

.venv/bin/python -m python.download_radarscenes_sample
.venv/bin/python -m python.run_radarscenes real_data/radarscenes_sequence_158
```

Plots and metrics are written to `real_output/`.

## Results

The real-data test uses sequence 158 from the
[RadarScenes dataset](https://radar-scenes.com/):

- 5 real targets produced confirmed tracks
- 1 false confirmed track
- 0 identity switches
- 0.37 m position RMSE

On an NVIDIA RTX 2000 Ada, CUDA became faster end to end at 1,024 objects. At
4,096 objects, the CUDA kernel was 125.6x faster than serial CPU; including data
transfer and allocation, the full operation was 1.34x faster.

See [the report](docs/REPORT.md), [real-data notes](docs/REAL_DATA.md), and
[benchmark record](docs/BENCHMARK_RECORD.md) for details and limitations.

This is an educational project, not a safety-critical radar system.
