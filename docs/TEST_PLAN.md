# Verification and Test Plan

## Goals

The test suite checks numerical correctness, tracker behavior, file-contract
stability, statistical consistency, determinism, and performance claims.
Accuracy thresholds should be justified by the configured process and sensor
noise, not tuned to one seed.

## Test levels

### Unit tests

| Component | Required cases |
|---|---|
| Motion model | `dt=0`, known constant velocity, covariance propagation, invalid negative `dt` |
| Linear KF | zero innovation, known hand-computed update, covariance symmetry, positive-semidefinite covariance |
| Radar model | axes/quadrants, finite-difference Jacobian, near-zero range, angle wrap across `-pi/pi` |
| Gating | inside/on/outside threshold, singular innovation covariance, incompatible dimensions |
| Association | one-to-one constraint, unassigned tracks/detections, tie policy, measurement-order determinism |
| Lifecycle | tentative birth, confirmation, coast, deletion, ID uniqueness |
| Sensor transforms | identity, translation/rotation, inverse/round trip, covariance frame rotation |

Numerical tests use combined absolute and relative tolerances. Covariance checks
explicitly verify finiteness, approximate symmetry, and nonnegative eigenvalues
within a small roundoff tolerance.

### Integration tests

1. A noiseless single target remains on the analytic trajectory.
2. A seeded noisy target produces a deterministic output file and improves
   position RMSE over raw Cartesianized measurements after initialization.
3. Two separated targets retain unique IDs with no clutter.
4. A crossing-target case records the baseline association behavior; it should
   not silently overwrite two tracks with one detection.
5. Clutter outside all gates does not update confirmed tracks.
6. A dropped-detection interval causes coasting then deletion at the configured
   boundary.
7. Two sensors with different timestamps are processed chronologically.
8. CPU serial, CPU parallel, and CUDA paths produce equivalent decisions and
   estimates within documented tolerance for identical inputs.

### Statistical tests

Run a fixed bank of seeds rather than drawing new seeds in CI. Evaluate RMSE,
NIS, and NEES over enough samples to expose bias. Statistical tests should use
broad model-derived acceptance bands to avoid flaky CI; tighter exploratory
charts can remain outside gating tests.

### Robustness tests

Reject or clearly handle malformed rows, non-finite values, duplicate/decreasing
timestamps (unless explicitly supported), empty measurement frames, very large
`dt`, near-singular covariance, and targets near the sensor origin. Run C++
tests under address/undefined-behavior sanitizers where supported.

## Performance protocol

Performance tests do not run as ordinary pass/fail unit tests. A benchmark run
must:

1. use a release build and fixed input data;
2. record hardware, software versions, backend, seed, and workload shape;
3. warm up before collecting samples;
4. exclude input generation from core timing but include a separate end-to-end
   measurement;
5. synchronize CUDA work before stopping an end-to-end timer;
6. retain every raw timing sample;
7. report median and tail latency plus `T_cpu/T_candidate`; and
8. validate output equivalence before interpreting speed.

The test should fail or be marked unavailable if a requested CUDA backend is
not actually active. It must never relabel CPU fallback timing as GPU timing.

## CSV artifact validation

`tests/validate_tracking_csv.py` is independent of the tracker API. It validates
a compact output contract and computes simple position error statistics.
Required columns are:

```text
timestamp, track_id, truth_x, truth_y, estimate_x, estimate_y
```

Rows must be finite; timestamps must be nondecreasing within each track. This
tool does not prove filter correctness, but it catches broken pipelines before
plotting or aggregate analysis.

The measurement interchange contract is independently checked by
`tests/validate_measurements_csv.py`. Its required columns are:

```text
timestamp, sensor_id, measurement_id, type, value1, value2, true_target_id,
sensor_x, sensor_y, range_std, bearing_std, position_std
```

`sensor_x` and `sensor_y` are mandatory finite world-frame coordinates on every
measurement row, including rows from a sensor at the origin. An omitted pose is
an invalid file rather than an implicit `(0, 0)` sensor. This matters because a
silent origin fallback changes the geometry of polar measurements.

Example:

```bash
python3 tests/validate_measurements_csv.py tests/fixtures/valid_measurements.csv
python3 tests/validate_tracking_csv.py tests/fixtures/valid_tracking.csv
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/probe_cuda.py > cuda_environment.json
```

The CUDA probe captures device/compiler availability only. It does not prove
that tracker execution used the GPU; that requires backend logging and runtime
evidence described in `docs/BENCHMARK_RECORD.md`.

## Acceptance checklist

- All deterministic unit/integration tests pass from a clean build.
- Required malformed-input cases fail with actionable diagnostics.
- Single-target RMSE and consistency criteria are declared before the final run.
- Multi-target reports include identity and false-track metrics.
- CPU parallel output agrees with serial output and speedup uses identical work.
- A real CUDA backend is exercised and measured on identified CUDA hardware; if
  unavailable, the project is explicitly incomplete with respect to GPU
  execution and reports only the fallback results.
- Generated figures and tables can be regenerated from committed code and
  archived raw data.
